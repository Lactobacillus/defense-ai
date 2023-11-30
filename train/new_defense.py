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
from data.dataset import VideoStage1Data
from train.util import LossAccumulator, EMA


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

    def build_wandb(self) -> None:  #WandB 초기화하고 로깅

        os.makedirs(os.path.join(self.args['result_path'], self.args['exp_name'], 'wandb'), exist_ok = True)

        wandb.init(name = self.args['exp_name'], project = 'defense', entity = self.args['wandb_entity'],
                    dir = os.path.join(self.args['result_path'], self.args['exp_name']),
                    config = self.args, config_exclude_keys = self.args['wandb_exclude'])

    def build_dataset(self) -> None:    # 비디오 데이터셋 초기화

        if 'split' in self.args['data_path']:

            self.train_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'], split = 'train')
            self.test_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'], split = 'valid')

        else:

            self.train_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'], split = 'train')
            self.test_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'], split = 'train')

    def build_model(self) -> None:  # CustomResNet50, Aggregator 모델 초기화 하고, EMA 적용

        self.model = CustomResNet50().to('cuda')
        self.model = torch.compile(self.model)

        self.aggr = Aggregator().to('cuda')
        self.aggr = torch.compile(self.aggr)


        self.model_ema = EMA(self.model, decay = 0.9)     # EMA (Exponential Moving Average) : 시계열 데이터의 추세를 부드럽게
        self.aggr_ema = EMA(self.aggr, decay = 0.9)           # 현재 값에 가중치를 부여하여 이동 평균을 계산


    def train(self,
            dataset: str = 'train',
            verbose: bool = False) -> None:

        self.model.train()
        self.model = self.model.to('cuda')  # to('cuda'): GPU 전송
        self.aggr.train()
        self.aggr = self.aggr.to('cuda')

        match dataset:

            case 'train': dset = self.train_data
            case 'valid': dset = self.test_data

        train_loader = DataLoader(dataset = dset,
            batch_size = self.args['batch_size'],   # 한번에 처리되는 데이터 샘플수 
            shuffle = True,
            drop_last = False,    # 데이터가 배치 크기로 나누어 떨어지지 않을 때 마지막 배치를 무시할지
            num_workers = 4,
            pin_memory = True)      # GPU로 데이터를 효율적으로 전송
        optimizer = torch.optim.AdamW(list(self.aggr.parameters()) + list(self.model.parameters()), lr = self.args['lr'])   # 가중치 감쇠를 통해 모델의 가중치 정규화
        grad_scaler = GradScaler()  # Mixed Precision Training 구현, 그래디언트 손실 방지

        for epoch in range(self.args['epoch']):

            self.model.train()
            self.aggr.train()
            epoch_start = timeit.default_timer()

            for idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):     # 훈련 데이터로부터 미니배치 가져옴

                video = batch['video'].to('cuda')
                label = batch['label'].bfloat16().unsqueeze(-1).to('cuda')# 데이터 전처리 (차원을 1로 확장)
                bs, fl, _, w, h = video.size()
                video = video.view(bs * fl, 3, w, h)

                with autocast(dtype = torch.bfloat16):   # Mixed Precision Training - FP16으로 수행

                    emb = self.model(video) # (bs * fl, dim, w, h)      # 순전파 수행
                    bsfl, d, w, h = emb.size()
                    emb = emb.view(bs, fl, d, w, h)

                    logit = self.aggr(emb)      # 로짓값 계산
                    loss = F.binary_cross_entropy_with_logits(logit, label)     # 손실값 계산

                optimizer.zero_grad(set_to_none = True)     # 역전파 수행
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()        # 가중치 업데이트


                if idx > 0 and idx % self.args['ema_update_freq'] == 0:         # EMA 업데이트

                    self.model_ema.update()
                    self.aggr_ema.update()

                if idx > 0 and idx % self.args['reset_freq'] == 0:

                    # self.aggr.reset_fc()
                    self.shrink_perturb()

                if self.use_wandb:

                    wandb.log({'train/stage1/loss': loss.item()})   # WandB 로그 기록

            self.save_checkpoint('latest')      # 체크포인트 저장
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

            print('[info] Epoch {} (Total: {}), elapsed time: {:.4f}'.format(epoch, self.args['epoch'], epoch_end - epoch_start))     # 경과시간 출력   

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

        if use_ema:     # EMA 적용

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

        new_model = CustomResNet50().to('cuda')
        new_aggr = Aggregator().to('cuda')

        for p1, p2 in zip(*[new_model.parameters(), self.model.parameters()]):
            
            p1.data = copy.deepcopy(shrink * p2.data + perturb * p1.data)

        for p1, p2 in zip(*[new_aggr.parameters(), self.aggr.parameters()]):
            
            p1.data = copy.deepcopy(shrink * p2.data + perturb * p1.data)

        self.model = new_model
        self.aggr = new_aggr

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

