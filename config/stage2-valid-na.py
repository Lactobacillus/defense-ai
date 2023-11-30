"""
memo
"""

import os
import sys
import getpass
from pathlib import Path

home = str(Path.home())
args = dict()

args['exp_name'] = 'stage2-valid-newarch'
args['result_path'] = os.path.join(home, 'temporary/')

args['data_path'] = os.path.join(home, 'dataset/only_face_split')
args['data_test_path'] = '/mnt/elice/dataset/test'
args['checkpoint_path'] = os.path.join(home, 'temporary/stage2-valid-newarch-fl16/')
args['frame_length'] = 16
args['batch_size'] = 12
args['distillation'] = 0.5

args['lr'] = 1e-4
args['epoch'] = 100
args['ema_update_freq'] = 50
args['reset_freq'] = 200

args['use_wandb'] = False
args['wandb_entity'] = 'lactobacillus_collabo'
args['wandb_exclude'] = ['use_wandb', 'wandb_entity', 'wandb_exclude', 'device', 'debug', 'result_path']

args['device'] = 'cuda'
args['debug'] = False
args['seed'] = None
