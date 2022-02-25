import torch
import logging

from . import checkpointsaver

LOGGER = logging.getLogger(__name__)

def create(conf, model, optimizer, scaler, architecture_conf):
    if conf['name'] == 'default_saver':
        architecture, name = architecture_conf['type'].split('/')
        conf['checkpoint_save_path'] = './' + architecture + '_' + name + '/'
        conf['top_save_path'] = conf['checkpoint_save_path'] + '/top/'
        saver = checkpointsaver.CheckpointSaver(conf, model, optimizer, scaler=scaler)
    else:
        raise AttributeError(f'not support saver config: {conf}')

    return saver