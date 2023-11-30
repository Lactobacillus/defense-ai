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

from model.model import CustomResNet50, LinearLayer
from data.dataset import VideoStage1Data, VideoPretrainData
from train.util import LossAccumulator
from sklearn.model_selection import train_test_split


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


        wandb.init(name = self.args['exp_name'], project = 'defense', entity = self.args['wandb_entity'],
                    dir = os.path.join(self.args['result_path'], self.args['exp_name']),
                    config = self.args, config_exclude_keys = self.args['wandb_exclude'])

    def build_dataset(self) -> None:
        
        # self.train_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'])
        # self.test_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'])
        full_data = VideoStage1Data(self.args['data_path'], self.args['frame_length'])
        train_size = int(0.8 * len(full_data))
        test_size = len(full_data) - train_size

        generator1 = torch.Generator().manual_seed(42)
        self.train_data, self.test_data = torch.utils.data.random_split(full_data, [train_size, test_size], generator=generator1)


    def build_model(self) -> None:

        self.processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
        self.model = CustomResNet50(pool=True).to('cuda')
        #self.model = torch.compile(self.model)

        self.linear = LinearLayer().to('cuda')
        #self.linear = torch.compile(self.model)

        self.model_ema = EMA(self.model, decay = 0.9)
        self.linear_ema = EMA(self.linear, decay = 0.9)

    def evaluate(self,
            dataset,
            use_ema: bool):
        
        self.model.eval()
        self.linear.eval()

        if use_ema:

            self.model_ema.apply_shadow()
            self.aggr_ema.apply_shadow()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # 그래디언트 계산 비활성화
            for batch in DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=False):
                video = batch['video'].to('cuda')
                label = batch['label'].float().unsqueeze(-1).to('cuda')
                bs, fl, _, w, h = video.size()
                video = video.view(bs * fl, 3, w, h)

                pixel = self.processor(video, return_tensors='pt').pixel_values.to('cuda')
                emb = self.model(pixel)
                logit = self.linear(emb)
                prob = torch.sigmoid(logit)

                """
                Voting
                """
                # final_prob = torch.mean(prob.view(bs, fl), dim=1)  # 평균으로 최종 결정
                # final_pred = (final_prob > 0.5).float()  # 최종 예측

                # # Loss 계산 (비디오 레벨)
                # loss = F.binary_cross_entropy(final_prob, label)
                # total_loss += loss.item()
                # correct_predictions += (final_pred == label).sum().item()
                # total_samples += bs

                """
                when not voting
                """
                loss = F.binary_cross_entropy_with_logits(logit, label)
                total_loss += loss.item() * video.size(0)
                predicted = (prob > 0.5).float()  # 예측된 클래스
                correct_predictions += (predicted == label).sum().item()
                total_samples += video.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        if use_ema:

            self.model_ema.restore()
            self.aggr_ema.restore()

        return avg_loss, accuracy

    def train(self,
            dataset: str = 'train',
            verbose: bool = False) -> None:

        self.model.train()
        self.model = self.model.to('cuda')
        self.linear.train()
        self.linear = self.linear.to('cuda')

        match dataset:

            case 'train': dset = self.train_data
            case 'test': dset = self.test_data

        train_loader = DataLoader(dataset = dset,
            batch_size = self.args['batch_size'],
            shuffle = True,
            drop_last = False,
            num_workers = 4,
            pin_memory = True)
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.linear.parameters()), lr = self.args['lr'])
        grad_scaler = GradScaler()

        # best_val_accuracy가 업데이트될때만 save checkpoint
        best_val_accuracy = 0.0
        for epoch in range(self.args['epoch']):
            # 이전 에포크의 체크포인트 로드
            # if epoch > 0:
            #     self.load_checkpoint('latest')
            self.model.train()
            self.linear.train()
            epoch_start = timeit.default_timer()

            for idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):

                video = batch['video'].to('cuda')
                label = batch['label'].float().unsqueeze(-1).to('cuda')
                bs, fl, _, w, h = video.size()
                #print('video.shape', video.shape) # [128, 1, 3, 128, 128]
                video = video.view(bs * fl, 3, w, h)

                with autocast(dtype = torch.float16):

                    pixel = self.processor(video, return_tensors = 'pt').pixel_values.to('cuda')
#                   print('pixel.shape', pixel.shape) # [1, 3, 244, 244]
                    emb = self.model(pixel) # (bs * fl, dim, w, h)
                    logit = self.linear(emb)

                    loss = F.binary_cross_entropy_with_logits(logit, label)

                optimizer.zero_grad(set_to_none = True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                if idx > 0 and idx % self.args['ema_update_freq'] == 0:

                    self.model_ema.update()
                    self.linear_ema.update()

                # if idx > 0 and idx % self.args['reset_freq'] == 0:

                    # self.linear.reset_layer()
                    # self.shrink_perturb()

                if self.use_wandb:
                    wandb.log({'train/stage1/loss': loss.item()})
            
            # 한 에포크 끝
            self.save_checkpoint('latest')
            self.save_checkpoint('epoch_{}'.format(epoch))
            
            # 훈련 및 검증 손실과 정확도 계산
            train_loss, train_accuracy = self.evaluate(self.train_data, False)
            val_loss, val_accuracy = self.evaluate(self.test_data, False)

            train_loss_ema, train_accuracy_ema = self.evaluate(self.train_data, True)
            val_loss_ema, val_accuracy_ema = self.evaluate(self.test_data, True)

            # 결과 출력
            print(f'Epoch {epoch}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
            print(f'Epoch {epoch}: Train Loss (ema): {train_loss_ema}, Train Accuracy (ema): {train_accuracy_ema}, Validation Loss (ema): {val_loss_ema}, Validation Accuracy (ema): {val_accuracy_ema}')

            # 검증 정확도가 향상될 경우 체크포인트 저장
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_checkpoint('best_checkpoint')

            # WandB 로그 업데이트
            if self.use_wandb:

                wandb.log({'train/loss': train_loss, 'train/accuracy': train_accuracy, 'validation/loss': val_loss, 'validation/accuracy': val_accuracy})
                wandb.log({'train/loss_ema': train_loss_ema, 'train/accuracy_ema': train_accuracy_ema, 'validation/loss_ema': val_loss_ema, 'validation/accuracy_ema': val_accuracy_ema})
      
            epoch_end = timeit.default_timer()

            print('[info] Epoch {} (Total: {}), elapsed time: {:.4f}'.format(epoch, self.args['epoch'], epoch_end - epoch_start))

        else:

            print('[info] Train finished')

    def save_checkpoint(self,
            filename: Optional[str] = None) -> None:

        if not filename: filename = 'checkpoint'

        checkpoint = {'model': self.model.cpu().state_dict(),
                      'linear': self.linear.cpu().state_dict(),
                      'model_ema': self.model_ema.state_dict(),
                      'linear_ema': self.linear_ema.state_dict(),
                    'args': self.args}
        
        self.model.to('cuda')
        self.linear.to('cuda')

        torch.save(checkpoint, os.path.join(self.result_root, '{}.pkl'.format(filename)))

    def load_checkpoint(self, filename: str) -> None:
        checkpoint_path = os.path.join(self.result_root, f'{filename}.pkl')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint '{checkpoint_path}' not found")
            return

        checkpoint = torch.load(checkpoint_path)

        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.model.to('cuda')

        if 'linear' in checkpoint:
            self.linear.load_state_dict(checkpoint['linear'])
            self.linear.to('cuda')

        if 'model_ema' in checkpoint:
            self.model_ema.load_state_dict(checkpoint['model_ema'])

        if 'linear_ema' in checkpoint:
            self.linear_ema.load_state_dict(checkpoint['linear_ema'])

        print(f"Loaded checkpoint '{checkpoint_path}'")

    def __del__(self) -> None:

        wandb.finish(0)