import os
import sys
import timeit
import wandb
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from transformers import AutoImageProcessor

from model.model import CustomResNet50, Aggregator
from data.dataset import VideoStage1Data
from train.util import LossAccumulator


class Stage1Trainer(object):

    def __init__(self,
            args: Dict[str, Any]) -> None:

        self.args = args
        self.result_root = os.path.join(args['result_path'], args['exp_name'])
        os.makedirs(self.result_root, exist_ok = True)

        self.build_model()
        self.build_dataset()

        if args['use_wandb'] and not args['debug']:

            self.use_wandb = True
            self.build_wandb()

        else: self.use_wandb = False

    def build_wandb(self) -> None:

        os.makedirs(os.path.join(self.args['result_path'], self.args['exp_name'], 'wandb'), exist_ok = True)

        wandb.init(name = name, project = 'defense-ai', entity = self.args['wandb_entity'],
                    dir = os.path.join(self.args['result_path'], name),
                    config = self.args, config_exclude_keys = self.args['wandb_exclude'])

    def build_dataset(self) -> None:

        self.train_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'])
        self.test_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'])

    def build_model(self) -> None:

        self.processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
        self.model = CustomResNet50().to('cuda')
        self.model = torch.compile(self.model)

        self.aggr = Aggregator().to('cuda')
        self.aggr = torch.compile(self.aggr)

    def train(self,
            dataset: str = 'train',
            verbose: bool = False) -> None:

        self.model.train()
        self.model = self.model.to('cuda')
        self.aggr.train()
        self.aggr = self.aggr.to('cuda')

        match dataset:

            case 'train': dset = self.train_data
            case 'test': dset = self.test_data

        train_loader = DataLoader(dataset = dset,
            batch_size = self.args['batch_size'],
            shuffle = True,
            drop_last = False,
            num_workers = 4,
            pin_memory = True)
        optimizer = torch.optim.AdamW(list(self.aggr.parameters()), lr = self.args['lr'])
        grad_scaler = GradScaler()

        for epoch in range(self.args['epoch']):

            self.model.train()
            self.aggr.train()
            epoch_start = timeit.default_timer()

            for idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):

                video = batch['video']
                label = batch['label'].float().unsqueeze(-1).to('cuda')
                bs, fl, _, w, h = video.size()
                video = video.view(bs * fl, 3, w, h)

                with autocast(dtype = torch.bfloat16):

                    with torch.no_grad():
                        
                        pixel = self.processor(video, return_tensors = 'pt').pixel_values.to('cuda')
                        emb = self.model(pixel) # (bs * fl, dim, w, h)
                        bsfl, d, w, h = emb.size()
                        emb = emb.view(bs, fl, d, w, h)

                    logit = self.aggr(emb)
                    loss = F.binary_cross_entropy_with_logits(logit, label)

                optimizer.zero_grad(set_to_none = True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                if self.use_wandb:

                    wandb.log({'train/pretrain': loss.item()})

            epoch_end = timeit.default_timer()

            print('[info] Epoch {} (Total: {}), elapsed time: {:.4f}'.format(epoch, self.args['epoch'], epoch_end - epoch_start))

        else:

            print('[info] Train finished')

    def save_checkpoint(self,
            filename: Optional[str] = None) -> None:

        if not filename: filename = 'checkpoint'

        checkpoint = {'model': self.model.cpu().state_dict(),
                    'args': self.args}

        torch.save(checkpoint, os.path.join(self.result_root, '{}.pkl'.format(filename)))

    def __del__(self) -> None:

        wandb.finish(0)