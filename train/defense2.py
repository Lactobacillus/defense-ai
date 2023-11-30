import os
import sys
import copy
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

from model.model import CustomResNet50, Aggregator
from data.dataset import VideoStage2Data
from train.util import LossAccumulator, EMA


class Stage2Trainer(object):

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

        wandb.init(name = self.args['exp_name'], project = 'defense', entity = self.args['wandb_entity'],
                    dir = os.path.join(self.args['result_path'], self.args['exp_name']),
                    config = self.args, config_exclude_keys = self.args['wandb_exclude'])

    def build_dataset(self) -> None:

        if 'split' in self.args['data_path']:

            self.train_data = VideoStage2Data(self.args['data_path'], self.args['frame_length'], split = 'train')
            self.test_data = VideoStage2Data(self.args['data_path'], self.args['frame_length'], split = 'valid')

        else:

            self.train_data = VideoStage2Data(self.args['data_path'], self.args['frame_length'], split = 'train')
            self.test_data = VideoStage2Data(self.args['data_path'], self.args['frame_length'], split = 'train')

    def build_model(self) -> None:

        self.model = CustomResNet50().to('cuda')
        self.aggr = Aggregator().to('cuda')

        self.model_ema = EMA(self.model, decay = 0.9)
        self.aggr_ema = EMA(self.aggr, decay = 0.9)

        loaded = torch.load(self.args['checkpoint_path'])

        self.model.load_state_dict(loaded['model'])
        self.aggr.load_state_dict(loaded['aggr'])
        self.model_ema.load_state_dict(loaded['model_ema'])
        self.aggr_ema.load_state_dict(loaded['aggr_ema'])

        #self.model = torch.compile(self.model)
        #self.aggr = torch.compile(self.aggr)

    def train(self,
            dataset: str = 'train',
            verbose: bool = False) -> None:

        self.model.train()
        self.model = self.model.to('cuda')
        self.aggr.train()
        self.aggr = self.aggr.to('cuda')

        match dataset:

            case 'train': dset = self.train_data
            case 'valid': dset = self.test_data

        train_loader = DataLoader(dataset = dset,
            batch_size = self.args['batch_size'],
            shuffle = True,
            drop_last = False,
            num_workers = 4,
            pin_memory = True)
        optimizer = torch.optim.AdamW(list(self.aggr.parameters()) + list(self.model.parameters()), lr = self.args['lr'])
        grad_scaler = GradScaler()

        for epoch in range(self.args['epoch']):

            self.model.train()
            self.aggr.train()
            epoch_start = timeit.default_timer()

            for idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):

                video = batch['video'].to('cuda')
                label = batch['label'].bfloat16().unsqueeze(-1).to('cuda')
                slabel = batch['label'].bfloat16().unsqueeze(-1).to('cuda')
                bs, fl, _, w, h = video.size()
                video = video.view(bs * fl, 3, w, h)

                with autocast(dtype = torch.bfloat16):

                    emb = self.model(video) # (bs * fl, dim, w, h)
                    bsfl, d, w, h = emb.size()
                    emb = emb.view(bs, fl, d, w, h)

                    logit = self.aggr(emb)
                    loss = F.binary_cross_entropy_with_logits(logit, label)
                    kd_loss = F.mse_loss(logit, slabel)

                    loss = (1 - args['distillation']) * loss + args['distillation'] * kd_loss

                optimizer.zero_grad(set_to_none = True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                if idx > 0 and idx % self.args['ema_update_freq'] == 0:

                    self.model_ema.update()
                    self.aggr_ema.update()

                #if idx > 0 and idx % self.args['reset_freq'] == 0:

                    # self.aggr.reset_fc()
                    #self.shrink_perturb()

                if self.use_wandb:

                    wandb.log({'train/stage1/loss': loss.item()})

            self.save_checkpoint('latest')
            self.save_checkpoint('epoch_{}'.format(epoch))

            if 'split' in self.args['data_path']:

                self.evaluate('train', True, 0.5, 5)
                self.evaluate('train', False, 0.5, 5)
                self.evaluate('valid', True, 0.5)
                self.evaluate('valid', False, 0.5)

            else:

                self.evaluate('train', True, 0.5, 5)
                self.evaluate('train', False, 0.5, 5)

            epoch_end = timeit.default_timer()

            print('[info] Epoch {} (Total: {}), elapsed time: {:.4f}'.format(epoch, self.args['epoch'], epoch_end - epoch_start))

        else:

            print('[info] Train finished')

    @torch.no_grad()
    def evaluate(self,
            dataset: str = 'train',
            use_ema: bool = True,
            threshold: float = 0.5,
            num_iter: int = 0) -> float:

        self.model.eval()
        self.model = self.model.to('cuda')
        self.aggr.eval()
        self.aggr = self.aggr.to('cuda')

        match dataset:

            case 'train': dset = self.train_data
            case 'valid': dset = self.test_data

        test_loader = DataLoader(dataset = dset,
            batch_size = self.args['batch_size'],
            shuffle = True,
            drop_last = False,
            num_workers = 4,
            pin_memory = True)

        if use_ema:

            self.model_ema.apply_shadow()
            self.aggr_ema.apply_shadow()

        correct = 0
        total = 0

        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):

            if num_iter > 0 and idx == num_iter:

                break

            video = batch['video'].to('cuda')
            label = batch['label'].float().unsqueeze(-1).to('cuda')
            bs, fl, _, w, h = video.size()
            video = video.view(bs * fl, 3, w, h)

            with autocast(dtype = torch.bfloat16):

                emb = self.model(video) # (bs * fl, dim, w, h)
                bsfl, d, w, h = emb.size()
                emb = emb.view(bs, fl, d, w, h)

                logit = self.aggr(emb)
        
            prob = torch.sigmoid(logit)
            pred = (prob > threshold).float()
            correct = correct + (pred == label).sum().item()
            total = total + label.size(0)

        acc = 100 * correct / total

        if self.use_wandb:

            if use_ema: wandb.log({'train/stage1_{}/acc_ema'.format(dataset): acc})
            else: wandb.log({'train/stage1_{}/acc'.format(dataset): acc})

        if use_ema:

            self.model_ema.restore()
            self.aggr_ema.restore()

        return acc

    def shrink_perturb(self,
            shrink: float = 0.9,
            perturb: float = 1e-3):

        for p in self.model.parameters():
            
            p.data = shrink * p + perturb * torch.randn_like(p)

        for p in self.aggr.parameters():
            
            p.data = shrink * p + perturb * torch.randn_like(p)

    def save_checkpoint(self,
            filename: Optional[str] = None) -> None:

        if not filename: filename = 'checkpoint'

        checkpoint = {'model': self.model.cpu().state_dict(),
                    'aggr': self.aggr.cpu().state_dict(),
                    'model_ema': self.model_ema.state_dict(),
                    'aggr_ema': self.aggr_ema.state_dict(),
                    'args': self.args}

        torch.save(checkpoint, os.path.join(self.result_root, '{}.pkl'.format(filename)))

    def __del__(self) -> None:

        wandb.finish(0)

