"""
memo
"""

import os
import sys
import getpass
from pathlib import Path

home = str(Path.home())
args = dict()

args['exp_name'] = 'stage1'
args['result_path'] = os.path.join(home, 'temporary/')

args['data_path'] = os.path.join(home, 'temporary/outputs_video')
args['frame_length'] = 16
args['batch_size'] = 16

args['lr'] = 1e-4
args['epoch'] = 100

args['use_wandb'] = True
args['wandb_entity'] = 'lactobacillus_collabo'
args['wandb_exclude'] = ['use_wandb', 'wandb_entity', 'wandb_exclude', 'device', 'debug', 'result_path']

args['device'] = 'cuda'
args['debug'] = False
args['seed'] = None
