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
args['data_path'] = './outputs_video'
args['frame_length'] = 1
args['batch_size'] = 128

args['lr'] = 1e-4
args['epoch'] = 100

args['ema_update_freq'] = 100
args['reset_freq'] = 200

args['use_wandb'] = True
args['wandb_entity'] = 'lactobacillus_collabo'
args['wandb_exclude'] = ['use_wandb', 'wandb_entity', 'wandb_exclude', 'device', 'debug', 'result_path']

args['device'] = 'cuda'
args['debug'] = False
args['seed'] = None
