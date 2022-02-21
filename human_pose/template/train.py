from curses.ascii import FF
import os
import sys
import logging
import datetime
import random
from tkinter import Y
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import copy
import argparse
from contextlib import suppress

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import hydra
from omegaconf import DictConfig, OmegaConf

import trainer

from tqdm import tnrange, tqdm 
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score
import itertools
import wandb
class Trainer():
    def __init__(self, conf, rank=0):
        self.conf = copy.deepcopy(conf)
        self.rank = rank
        self.is_master = True if rank == 0 else False
        self.writer = SummaryWriter()
        self.set_env()
        
    def set_env(self):
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(self.rank)
        if self.conf.base.wandb is True:
                    config={
                    "architecture": self.conf.architecture.type,
                    "optimizer": self.conf.optimizer.type,
                    "scheduler": self.conf.scheduler.type,
                    "epochs": self.conf.hyperparameter.epochs,
                    "batch_size": self.conf.dataset.train.batch_size,
                    "loss": self.conf.loss.type,
                    "learning rate": self.conf.optimizer.params.lr}
                    if self.conf.optimizer.params.weight_decay > 0:
                        config['weight_decay'] = self.conf.optimizer.params.weight_decay
                    wandb.init(project="action_recognition_with_transformer", config=config)
                    wandb.run.name = self.conf.architecture.type
                    wandb.run.save()

        # mixed precision
        self.amp_autocast = suppress
        if self.conf.base.use_amp is True:
            self.amp_autocast = torch.cuda.amp.autocast
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            
            if self.is_master:
                print(f'[Hyper]: Use Mixed precision - float16')
        else:
            self.scaler = None
        
        # Scheduler
        if self.conf.scheduler.params.get('T_max', None) is None:
            self.conf.scheduler.params.T_max = self.conf.hyperparameter.epochs
        
        self.start_epoch = 1
    def build_looger(self, is_use:bool):
        if is_use == True: 
            logger = trainer.log.create(self.conf)
            return logger
        else: 
            pass

    def build_model(self, num_classes=-1):
        model = trainer.architecture.create(self.conf.architecture)
        model = model.to(device=self.rank, non_blocking=True)
        model = DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)

        return model

    def build_optimizer(self, model):
        optimizer = trainer.optimizer.create(self.conf.optimizer, model)
        return optimizer


    def build_scheduler(self, optimizer):
        scheduler = trainer.scheduler.create(self.conf.scheduler, optimizer)

        return scheduler
    # TODO: modulizaing
    def build_dataloader(self, ):

        train_loader, train_sampler, valid_loader, valid_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'train')

        test_loader, test_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'test')

        return train_loader, train_sampler, valid_loader, valid_sampler, test_loader, test_sampler

    def build_loss(self):
        criterion = trainer.loss.create(self.conf.loss, self.rank)
        criterion.to(device=self.rank, non_blocking=True)

        return criterion

    def build_saver(self, model, optimizer, scaler):
        saver = trainer.saver.create(self.conf.saver, model, optimizer, scaler)

        return saver
    
    def load_model(self, model, path):
        data = torch.load(path)
        key = 'model' if 'model' in data else 'state_dict'

        if not isinstance(model, (DataParallel, DDP)):
            model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
        return model
    
    def train_one_epoch(self, epoch, model, dl, criterion, optimizer,logger):
        # for step, (image, label) in tqdm(enumerate(dl), total=len(dl), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        train_hit = 0
        train_total = 0
        one_epoch_loss = 0
        # eval_result = [accuracy, precision, recall, loss, image_num]
        t_loss = 0
        t_imgnum = 0
        accuracy = 0
        accuracy_from_scikit_learn = 0
        recall = 0
        precision = 0
        TN = 0
        TP = 0
        FN = 0
        FP = 0

        #torch.set_default_tensor_type(torch.cuda.LONG)
        model.train()
        pbar = tqdm(
            enumerate(dl), 
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl), 
            desc=f"train:{epoch}/{self.conf.hyperparameter.epochs}", 
            disable=not self.is_master
            )
        current_step = epoch
        for step, (image, label) in pbar:
            #image= torch.stack(image)
            #label= torch.stack(label)
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input)
                
                y_pred = y_pred.to(torch.float)
                label = label.to(torch.long)
                loss = criterion(y_pred, label).float()
                #loss.requires_grad = True
            optimizer.zero_grad()
            
            if self.scaler is None:
                loss.backward()
                optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            #self.plot_grad_flow(model.named_parameters())
            y_pred = y_pred.detach().cpu().numpy()

            label = label.detach().cpu().numpy()
            t_imgnum += y_pred.shape[0]
            t_loss += (loss.item() * y_pred.shape[0])

            accuracy, recall, precision, TN, FN, TP, FP = self.evaluation(y_pred, label, TN, FN, TP, FP)

            y_pred = np.argmax(y_pred, axis=1)
            y_pred = np.around(y_pred)
            accuracy_from_scikit_learn += (accuracy_score(label,y_pred) * y_pred.shape[0])

            if step % 100 == 0:
                accuracy_sci = accuracy_from_scikit_learn / t_imgnum
                pbar.set_postfix({'train_Acc':accuracy, 'train_acc2':accuracy_sci,'recall':recall, 'precision':precision, 'train_Loss':round(loss.item(),2)})
        
        #torch.distributed.reduce(counter, 0)
        if self.is_master:
            accuracy = (TP + TN) / (TP + TN + FP + FN + 0.000001)
            recall = (TP) / (TP + FN + 0.000001)
            precision = (TP) / (TP + FP + 0.000001)
            metric = {'Acc': accuracy, 'Loss': t_loss / t_imgnum,'optimizer':optimizer}
            self.writer.add_scalar("Loss/train", t_loss / t_imgnum, epoch)
            self.writer.add_scalar("ACC/train", accuracy, epoch)
            self.writer.add_scalar('Recall/train', recall, epoch)
            self.writer.add_scalar('Precision/train', precision, epoch)
            self.writer.flush()
            if self.conf.base.wandb is True:
                wandb.log({
        "train precision": precision,
        "train recall": recall})
        # return loss, accuracy
        #return t_loss / t_imgnum, t_acc / t_imgnum, t_iou / t_imgnum, dl
        return t_loss/ t_imgnum, accuracy, accuracy_from_scikit_learn/t_imgnum, dl


    @torch.no_grad()
    def eval(self, epoch, model, dl, criterion,logger):
        # 0: val_loss, 1: val_hit, 2: val_total, 3: len(dl)
        counter = torch.zeros((4, ), device=self.rank)
        model.eval()
        pbar = tqdm(
            enumerate(dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl),
            desc=f"val  :{epoch}/{self.conf.hyperparameter.epochs}", 
            disable=not self.is_master
            ) # set progress bar
        current_step = epoch
        t_loss = 0
        t_imgnum = 0
        accuracy = 0
        recall = 0
        precision = 0
        TN = 0
        TP = 0
        FN = 0
        FP = 0

        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input)
                y_pred = y_pred.to(torch.float)
                label = label.to(torch.long)
                loss = criterion(y_pred, label).float()
            y_pred = y_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            t_imgnum += y_pred.shape[0]
            t_loss += (loss.item() * y_pred.shape[0])

            accuracy, recall, precision, TN, FN, TP, FP = self.evaluation(y_pred, label, TN, FN, TP, FP)

            if step % 100 == 0:
                pbar.set_postfix({'Valid_Acc':accuracy, 'recall':recall, 'precision':precision, 'valid_Loss':round(loss.item(),2) / t_imgnum})
        
        #torch.distributed.reduce(counter, 0)
        if self.is_master:
            accuracy = (TP + TN) / (TP + TN + FP + FN + 0.000001)
            recall = (TP) / (TP + FN + 0.000001)
            precision = (TP) / (TP + FP + 0.000001)
            self.writer.add_scalar("Loss/val", t_loss / t_imgnum, epoch)
            self.writer.add_scalar("ACC/val", accuracy, epoch)
            self.writer.add_scalar('Recall/val', recall, epoch)
            self.writer.add_scalar('Precision/val', precision, epoch)
            self.writer.flush()
            if self.conf.base.wandb is True:
                wandb.log({
        "valid precision": precision,
        "valid recall": recall})
        # return loss, accuracy
        #return t_loss / t_imgnum, t_acc / t_imgnum, t_iou / t_imgnum, dl
        return t_loss/ t_imgnum, accuracy, dl

    def train_eval(self):
        model = self.build_model()
        if self.conf.base.wandb is True:
            wandb.watch(model)
        criterion = self.build_loss()
        optimizer = self.build_optimizer(model)

        scheduler = self.build_scheduler(optimizer)
        train_dl, train_sampler,valid_dl, valid_sampler, test_dl, test_sampler= self.build_dataloader()

        logger = self.build_looger(is_use=self.is_master)
        saver = self.build_saver(model, optimizer, self.scaler)
        # Wrap the model
        
        # initialize
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

        # add graph to tensorboard
        #if logger is not None:
            #logger.update_graph(model, torch.rand((1,1,28,28)).float())

        # load checkpoint
        if self.conf.base.resume == True:
            self.start_epoch = saver.load_for_training(model,optimizer,self.rank,scaler=None)
        
        for epoch in range(self.start_epoch, self.conf.hyperparameter.epochs + 1):
            train_sampler.set_epoch(epoch)
            # train
            train_loss, train_acc, acc2, train_dl = self.train_one_epoch(epoch, model, train_dl, criterion, optimizer, logger)
            scheduler.step()

            # eval
            valid_loss, valid_acc, valid_dl = self.eval(epoch, model, valid_dl, criterion, logger)
            
            torch.cuda.synchronize()

            # save_model
            saver.save_checkpoint(epoch=epoch, model=model, loss=train_loss, rank=self.rank, metric=valid_acc)

            if self.is_master:
                print(f'Epoch {epoch}/{self.conf.hyperparameter.epochs} - train_Acc: {train_acc:.3f}, train_ACC2: {acc2:.3f}, train_Loss: {train_loss:.3f}, valid_Acc: {valid_acc:.3f}, valid_Loss: {valid_loss:.3f}')
                if self.conf.base.wandb is True:
                    wandb.log({
            "train accuracy": train_acc,
            "train_loss": train_loss,
            "valid accuracy":  valid_acc,
            "valid_loss": valid_loss})

    def test_sample_visualization(self, y_pred, label, num, thresh=0.5):
        y_pred = y_pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()[0]
        y_pred[y_pred > thresh] = 1
        y_pred[y_pred <= thresh] = 0
        #img_grid = torchvision.utils.make_grid(y_pred)
        #label_grid = torchvision.utils.make_grid(label)
        self.writer.add_image(str(num) + '/prediction', y_pred, global_step=25, dataformats='HW')
        self.writer.add_image(str(num) + '/label', label, global_step=25, dataformats='HW')
        self.writer.flush()




    def test(self):
        # settings
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        saver = self.build_saver(model, optimizer, self.scaler)
        checkpoint_path = '/media/ddl/새 볼륨/Git/human_pose_estimation/human_pose/outputs/2022-02-21/22-28-08/checkpoint/top/001st_checkpoint_epoch_77.pth.tar'
        saver.load_for_inference(model, self.rank, checkpoint_path)
        train_dl, train_sampler,valid_dl, valid_sampler, test_dl, test_sampler= self.build_dataloader()
        # inference
        pbar = tqdm(
            enumerate(test_dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(test_dl),
            disable=not self.is_master
            ) # set progress bar
        t_loss = 0
        t_imgnum = 0
        accuracy = 0
        recall = 0
        precision = 0
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        epoch = 1

        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input).squeeze()
                label = label.to(torch.int64)
            y_pred = y_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            t_imgnum += y_pred.shape[0]
            accuracy, recall, precision, TN, FN, TP, FP = self.evaluation(y_pred, label, TN, FN, TP, FP)
        if self.is_master:
            self.writer.add_scalar("ACC/test", accuracy, epoch)
            self.writer.add_scalar('Precision/test', precision, epoch)
            self.writer.add_scalar('Recall/test', recall, epoch)
            self.writer.flush()
            
        return accuracy, recall, precision

    def inference(self):
        # settings
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        saver = self.build_saver(model, optimizer, self.scaler)
        checkpoint_path = '/home/ddl/git/human_pose_estimation/human_pose/outputs/2022-02-21/14-38-04/checkpoint/top/001st_checkpoint_epoch_98.pth.tar'
        saver.load_for_inference(model, self.rank, checkpoint_path)
        train_dl, train_sampler,valid_dl, valid_sampler, test_dl, test_sampler= self.build_dataloader()
        # inference
        pbar = tqdm(
            enumerate(test_dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(test_dl),
            disable=not self.is_master
            ) # set progress bar
        t_loss = 0
        t_imgnum = 0
        accuracy = 0
        recall = 0
        precision = 0
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        epoch = 1

        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input).squeeze()
                label = label.to(torch.int64)
            y_pred = y_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            t_imgnum += y_pred.shape[0]
            accuracy, recall, precision, TN, FN, TP, FP = self.evaluation(y_pred, label, TN, FN, TP, FP)
        if self.is_master:
            self.writer.add_scalar("ACC/test", accuracy, epoch)
            self.writer.add_scalar('Precision/test', precision, epoch)
            self.writer.add_scalar('Recall/test', recall, epoch)
            self.writer.flush()
            
        return accuracy, recall, precision

    
    def evaluation(self, y_pred: np.array, label: np.array, TN, FN, TP ,FP):
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = np.around(y_pred)
        for i in range(len(y_pred)):
            if y_pred[i] == 0 and label[i] == 0:
                TN += 1
            elif y_pred[i] == 0 and label[i] >= 1:
                FN += 1
            elif y_pred[i] == label[i]:
                TP += 1
            else:
                FP += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN + 0.000001)
        recall = (TP) / (TP + FN + 0.000001)
        precision = (TP) / (TP + FP + 0.000001)
        
        return accuracy, recall, precision, TN, FN, TP, FP



    def run(self):
        if self.conf.base.mode == 'train':
            pass
        elif self.conf.base.mode == 'train_eval':
            self.train_eval()
        elif self.conf.base.mode == 'finetuning':
            pass
        elif self.conf.base.mode == 'test':
            test_acc, test_recall, test_precision = self.test()
            print('test_acc:',test_acc, 'test_precision', test_precision, 'test_recall', test_recall)



def set_seed(conf):
    if conf.base.seed is not None:
        conf.base.seed = int(conf.base.seed, 0)
        print(f'[Seed] :{conf.base.seed}')
        os.environ['PYTHONHASHSEED'] = str(conf.base.seed)
        random.seed(conf.base.seed)
        np.random.seed(conf.base.seed)
        torch.manual_seed(conf.base.seed)
        torch.cuda.manual_seed(conf.base.seed)
        #torch.cuda.manual_seed_all(conf.base.seed)  # if use multi-G
        torch.backends.cudnn.deterministic = True


def runner(rank, conf):
    # Set Seed
    set_seed(conf)

    os.environ['MASTER_ADDR'] = conf.MASTER_ADDR
    os.environ['MASTER_PORT'] = conf.MASTER_PORT

    print(f'Starting train method on rank: {rank}')
    dist.init_process_group(
        backend='nccl', world_size=conf.base.world_size, init_method='env://',
        rank=rank
    )
    trainer = Trainer(conf, rank)
    trainer.run()



@hydra.main(config_path='conf', config_name='mine')
def main(conf: DictConfig) -> None:
    print(f'Configuration\n{OmegaConf.to_yaml(conf)}')
    
    mp.spawn(runner, nprocs=conf.base.world_size, args=(conf, ))
    

if __name__ == '__main__':
    main()
