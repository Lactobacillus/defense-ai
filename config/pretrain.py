"""
memo
"""

import os
import sys
import getpass
from pathlib import Path

home = str(Path.home())
args = dict()

args['exp_name'] = 'example'
args['result_path'] = os.path.join(home, 'temporary/')

args['use_wandb'] = False
args['wandb_entity'] = ''
args['wandb_exclude'] = ['use_wandb', 'wandb_entity', 'wandb_exclude', 'device', 'debug', 'result_path']

args['device'] = 'cuda'
args['debug'] = True
args['seed'] = None
