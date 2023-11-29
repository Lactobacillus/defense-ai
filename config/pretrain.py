"""
memo
"""

import os
import sys
import getpass
from pathlib import Path

home = str(Path.home())
args = dict()

args['exp_name'] = 'pretrain'
args['result_path'] = os.path.join(home, 'temporary/')

args['data_path'] = '/mnt/elice/dataset/train/real'
args['frame_length'] = 16
args['batch_size'] = 4

args['lr'] = 1e-4
args['epoch'] = 100

args['use_wandb'] = False
args['wandb_entity'] = 'defense'
args['wandb_exclude'] = ['use_wandb', 'wandb_entity', 'wandb_exclude', 'device', 'debug', 'result_path']

args['device'] = 'cuda'
args['debug'] = True
args['seed'] = None
