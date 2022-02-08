from curses.ascii import FF
import os
import sys
import logging
import datetime
import random
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

        train_loader, train_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'train')

        valid_loader, valid_sampler = trainer.new_dataset.create(conf = self.conf.dataset,
        world_size=self.conf.base.world_size,
        local_rank=self.rank,
        mode = 'valid')

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
        t_acc = np.zeros(1)
        t_iou = np.zeros(1)
        t_loss = np.zeros(1)
        t_imgnum = np.zeros(1)

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
                #if len(y_pred.shape) == 4:
                #    y_pred = torch.argmax(y_pred, dim=1)
                #else:
                #    y_pred = torch.argmax(y_pred, dim=0)
                if len(y_pred.shape) != len(label.shape):
                    if len(y_pred.shape) > len(label.shape):
                        label = torch.unsqueeze(label, dim=1)
                    else:
                        y_pred = torch.unsqueeze(y_pred, dim=1)
                #y_pred = torch.argmax(y_pred, dim=1)[None,:]
                #label = F.one_hot(label.to(torch.int64), num_classes=10)
                y_pred = y_pred.to(torch.float)
                label = label.to(torch.int64)
                loss = criterion(y_pred, label).float()
                #loss.requires_grad = True
            optimizer.zero_grad(set_to_none=True)
            t_imgnum += y_pred[0]
            
            if self.scaler is None:
                loss.backward()
                optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            y_pred = y_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            y_pred = np.around(y_pred)
            TN = 0
            TP = 0
            FN = 0
            FP = 0
            for i in range(len(y_pred)):
                if y_pred[i] == 0 and label == 0:
                    TN += 1
                elif y_pred[i] == 0 and label >= 1:
                    FN += 1
                elif y_pred[i] >= 1 and label == 0:
                    FP += 1
                elif y_pred[i] == 1 and label == 1:
                    TP += 1
                elif y_pred[i] == 2 and label == 2:
                    TP += 1
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            recall = (TP) / (TP + FN)
            precision = (TP) / (TP + FP)

            if step % 100 == 0:
                pbar.set_postfix({'train_Acc':accuracy, 'recall':recall, 'precision':precision, 'train_Loss':round(loss.item(),2)})
        
        #torch.distributed.reduce(counter, 0)
        if self.is_master:
            metric = {'Acc': accuracy, 'Loss': t_loss / t_imgnum,'optimizer':optimizer}
            self.writer.add_scalar("Loss/train", t_loss / t_imgnum, epoch)
            self.writer.add_scalar("ACC/train", accuracy, epoch)
            self.writer.add_scalar('Recall/train', recall/t_imgnum)
            self.writer.add_scalar('Precision/train', precision/t_imgnum)
            logger.update_log(metric,current_step,'train') # update logger step
            logger.update_histogram(model,current_step,'train') # update weight histogram 
            logger.update_image(image,current_step,'train') # update transpose image
            logger.update_metric()
            self.writer.flush()
        # return loss, accuracy
        #return t_loss / t_imgnum, t_acc / t_imgnum, t_iou / t_imgnum, dl
        return t_loss/ t_imgnum, accuracy, dl


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
        t_acc = np.zeros(1)
        t_iou = np.zeros(1)
        t_loss = np.zeros(1)
        t_imgnum = np.zeros(1)

        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input)
                if len(y_pred.shape) != len(label.shape):
                    if len(y_pred.shape) > len(label.shape):
                        label = torch.unsqueeze(label, dim=1)
                    else:
                        y_pred = torch.unsqueeze(y_pred, dim=1)
                y_pred = y_pred.to(torch.float)
                label = label.to(torch.int64)
                loss = criterion(y_pred, label).float()
                #loss.requires_grad = False
            y_pred = y_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            y_pred = np.around(y_pred)
            TN = 0
            TP = 0
            FN = 0
            FP = 0
            for i in range(len(y_pred)):
                if y_pred[i] == 0 and label == 0:
                    TN += 1
                elif y_pred[i] == 0 and label >= 1:
                    FN += 1
                elif y_pred[i] >= 1 and label == 0:
                    FP += 1
                elif y_pred[i] == 1 and label == 1:
                    TP += 1
                elif y_pred[i] == 2 and label == 2:
                    TP += 1
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            recall = (TP) / (TP + FN)
            precision = (TP) / (TP + FP)

            if step % 100 == 0:
                pbar.set_postfix({'train_Acc':accuracy, 'recall':recall, 'precision':precision, 'train_Loss':round(loss.item(),2)})
        
        #torch.distributed.reduce(counter, 0)
        if self.is_master:
            metric = {'Acc': accuracy, 'Loss': t_loss / t_imgnum,'optimizer':optimizer}
            self.writer.add_scalar("Loss/train", t_loss / t_imgnum, epoch)
            self.writer.add_scalar("ACC/train", accuracy, epoch)
            self.writer.add_scalar('Recall/train', recall/t_imgnum)
            self.writer.add_scalar('Precision/train', precision/t_imgnum)
            logger.update_log(metric,current_step,'train') # update logger step
            logger.update_histogram(model,current_step,'train') # update weight histogram 
            logger.update_image(image,current_step,'train') # update transpose image
            logger.update_metric()
            self.writer.flush()
        # return loss, accuracy
        #return t_loss / t_imgnum, t_acc / t_imgnum, t_iou / t_imgnum, dl
        return t_loss/ t_imgnum, accuracy, dl


    def train_eval(self):
        model = self.build_model()
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
            train_loss, train_acc, train_iou, train_dl = self.train_one_epoch(epoch, model, train_dl, criterion, optimizer, logger)
            scheduler.step()

            # eval
            valid_loss, valid_acc, valid_iou = self.eval(epoch, model, valid_dl, criterion, logger)
            
            torch.cuda.synchronize()

            # save_model
            saver.save_checkpoint(epoch=epoch, model=model, loss=train_loss, rank=self.rank, metric=valid_acc)

            if self.is_master:
                print(f'Epoch {epoch}/{self.conf.hyperparameter.epochs} - train_Acc: {train_acc[0]:.3f}, train_Loss: {train_loss[0]:.3f}, valid_Acc: {valid_acc[0]:.3f}, valid_Loss: {valid_loss[0]:.3f}')

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
        checkpoint_path = '/home/ddl/git/template/outputs/save/1_class_segmentation/Unet_efficientnet_B2_lr_0.0001]/Semantic_segmentation_class_1_training_with_complicate_augmentation(5-fold)/checkpoint/top/001st_checkpoint_epoch_158.pth.tar'
        saver.load_for_inference(model, self.rank, checkpoint_path)
        train_dl, train_sampler,valid_dl, valid_sampler, test_dl, test_sampler= self.build_dataloader()
        # inference
        pbar = tqdm(
            enumerate(test_dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(test_dl),
            disable=not self.is_master
            ) # set progress bar
        t_acc = np.zeros(1)
        t_precision = np.zeros(1)
        t_iou = np.zeros(1)
        t_imgnum = np.zeros(1)
        epoch = 1

        for step, (image, label) in pbar:
            image = image.to(device=self.rank, non_blocking=True).float()
            label = label.to(device=self.rank, non_blocking=True).float()
            with self.amp_autocast():
                input = image
                y_pred = model(input).squeeze()
                label = label.to(torch.int64)
            self.test_sample_visualization(y_pred, label, step)
            accuracies, ious = self.evaluation_for_semantic_segmentation(y_pred, label)
            temp_acc, temp_iou, temp_imgnum = np.zeros(1), np.zeros(1), np.zeros(1)
            for i in range(image.shape[0]):
                temp_acc += accuracies[i]
                temp_iou += ious[i]
                t_acc += accuracies[i]
                t_iou += ious[i]
            t_imgnum += image.shape[0]
            temp_imgnum += image.shape[0]
        if self.is_master:
            self.writer.add_scalar("ACC/test", t_acc / t_imgnum, epoch)
            self.writer.add_scalar('IoU/test', t_iou / t_imgnum)
            self.writer.flush()
            
        return t_acc / t_imgnum, t_iou / t_imgnum




    def run(self):
        if self.conf.base.mode == 'train':
            pass
        elif self.conf.base.mode == 'train_eval':
            self.train_eval()
        elif self.conf.base.mode == 'finetuning':
            pass
        elif self.conf.base.mode == 'test':
            test_acc, test_iou = self.test()
            print('test_acc:',test_acc, 'test_iou', test_iou)


def set_seed(conf):
    if conf.base.seed is not None:
        conf.base.seed = int(conf.base.seed, 0)
        print(f'[Seed] :{conf.base.seed}')
        os.environ['PYTHONHASHSEED'] = str(conf.base.seed)
        random.seed(conf.base.seed)
        np.random.seed(conf.base.seed)
        torch.manual_seed(conf.base.seed)
        torch.cuda.manual_seed(conf.base.seed)
        torch.cuda.manual_seed_all(conf.base.seed)  # if use multi-G
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

