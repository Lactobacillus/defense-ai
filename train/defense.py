import os
import sys
import timeit
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.model import TempModel
from data.dataset import SingleImage
from train.util import LossAccumulator


class Trainer(object):

    def __init__(self,
            args: Dict[str, Any]) -> None:

        self.args = args
        self.result_root = os.path.join(args['result_path'], args['exp_name'])
        os.makedirs(self.result_root, exist_ok = True)

        self.build_model()
        self.build_dataset()

    def build_dataset(self) -> None:

        self.train_data = TempModel(self.args['data_path'])
        self.test_data = TempModel(self.args['data_path'])

    def build_model(self) -> None:

        self.model = TempModel(self.args)

    def train(self,
            dataset: str = 'train',
            verbose: bool = False) -> None:

        self.model.train()
        self.model = self.model.to('cuda')

        match dataset:

            case 'train': dset = self.train_data
            case 'test': dset = self.test_data

        train_loader = DataLoader(dataset = dset,
            batch_size = self.args['batch_size'],
            shuffle = True,
            drop_last = False,
            num_workers = 2,
            pin_memory = True)
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr = self.args['lr'])

        for epoch in range(self.args['epoch']):

            self.model.train()
            epoch_start = timeit.default_timer()

            for idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):

                batch = {k: batch[k].to('cuda') for k in batch}

                pred = self.model(batch['coord'])
                true = batch['pixel']
                loss = F.mse_loss(pred, true)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_end = timeit.default_timer()

            if epoch % self.args['test_freq'] == 0:
                
                print('\n[info] Epoch {} (Total: {})'.format(epoch, self.args['epoch']))
                self.test(epoch, 'train')
                self.save_image('result_{}'.format(epoch))

        else:

            print('[info] Train finished')

    @torch.no_grad()
    def test(self,
            epoch: Union[int, str],
            dataset: str = 'test') -> None:

        loss_meter = LossAccumulator()

        self.model.eval()
        self.model = self.model.to('cuda')

        match dataset:

            case 'train': dset = self.train_data
            case 'test': dset = self.test_data

        test_loader = DataLoader(dataset = dset,
            batch_size = self.args['batch_size'],
            shuffle = False,
            drop_last = False,
            num_workers = 2,
            pin_memory = True)

        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):

            batch = {k: batch[k].to('cuda') for k in batch}

            pred = self.model(batch['coord'])
            true = batch['pixel']
            loss = F.mse_loss(pred, true)

            loss_meter.add({'loss': loss}, true.size(0))

        total_loss = loss_meter.get()['loss'].item()

        print('[info] {} data loss: {:.4f} (epoch: {})'.format(dataset, total_loss, epoch))

        return total_loss

    def save_checkpoint(self,
            filename: Optional[str] = None) -> None:

        if not filename: filename = 'checkpoint'

        checkpoint = {'model': self.model.cpu().state_dict(),
                    'args': self.args}

        torch.save(checkpoint, os.path.join(self.result_root, '{}.pkl'.format(filename)))
