"""
memo
"""

import os
import sys
import getpass
from pathlib import Path

home = str(Path.home())
args = dict()

args['exp_name'] = 'finetune'
args['result_path'] = os.path.join(home, 'temporary_reset/')

# args['data_path'] = '/mnt/elice/dataset'
args['data_path'] = os.path.join(home, 'dataset/only_face')
args['checkpoints_path'] = '/home/elicer/temporary'
args['data_test_path'] = '/mnt/elice/dataset/test'

args['frame_length'] = 64
args['batch_size'] = 4

args['lr'] = 1e-4
args['epoch'] = 100

args['use_wandb'] = True
args['wandb_entity'] = 'lactobacillus_collabo'
args['wandb_exclude'] = ['use_wandb', 'wandb_entity', 'wandb_exclude', 'device', 'debug', 'result_path']

args['device'] = 'cuda'
args['debug'] = False
args['seed'] = None
