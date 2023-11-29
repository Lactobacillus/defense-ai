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

from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining, VideoMAEForVideoClassification
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from train.util import LossAccumulator


class PreTrainer(object):

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

        self.train_data = TempModel(self.args['data_path'])
        self.test_data = TempModel(self.args['data_path'])

    def build_model(self) -> None:

        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base").to('cuda')
        self.model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base").to('cuda')
        # self.peft_conf = LoraConfig(inference_mode = False,
        #                     r = 8,
        #                     lora_alpha = 32,
        #                     lora_dropout = 0.1,
        #                     task_type = TaskType.SEQ_CLS,
        #                     target_modules = ['query', 'key', 'value'])
        # self.model = get_peft_model(self.model, self.peft_config)
        # self.model.print_trainable_parameters()

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
        optimizer = torch.optim.AdamW(list(self.model.parameters()), lr = self.args['lr'])
        grad_scaler = GradScaler()
        num_patches_per_frame = (self.model.config.image_size // self.model.config.patch_size) ** 2
        seq_length = (num_frames // self.model.config.tubelet_size) * num_patches_per_frame

        for epoch in range(self.args['epoch']):

            self.model.train()
            epoch_start = timeit.default_timer()

            for idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):

                batch = {k: batch[k].to('cuda') for k in batch}

                with autocast(dtype = torch.bfloat16):

                    raise NotImplementedError

                    inputs = processor(video, return_tensors = 'pt').pixel_values.to('cuda')
                    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

                    outputs = self.model(pixel_values, bool_masked_pos = bool_masked_pos)
                    loss = outputs.loss

                optimizer.zero_grad(set_to_none = True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
