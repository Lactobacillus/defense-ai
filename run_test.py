import os
import sys
import codecs
import argparse
import torch
import numpy as np
import pandas as pd
import random
import pickle
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from data.preprocess import Preprocess
from model.model import CustomResNet50, Aggregator
from data.dataset import VideoStage1Data
from train.util import LossAccumulator, EMA

def make_video_tensor(video_path, frame_length) -> torch.Tensor:
    fn = video_path
    video = video2tensor(fn)

    start = random.randrange(0, video.size(0) - frame_length - 1)
    end = start + frame_length
    video = video[start:end, ...]

    return video

def video2tensor(
        filename: str,
        output_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:

    video, _, _ = tv.io.read_video(filename, output_format = 'TCHW', pts_unit = 'sec')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    video = torch.stack([transform(frame) for frame in video])

    return video

def main(args: Dict[str, Any],
        checkpoint_file_name: int,
        use_ema:bool,
        name: str,
        threshold: float = 0.5) -> None:

    checkpoint_file_name = checkpoint_file_name.replace('.pkl', '')
    checkpoint_file_path = os.path.join(args['result_path'], args['exp_name'], f'{checkpoint_file_name}.pkl')
    checkpoint = torch.load(checkpoint_file_path)

    submission = pd.read_csv('/home/elicer/sample_submission.csv')
    test_file_names = submission['path'].tolist()

    # 모델 불러오기
    model = CustomResNet50().to('cuda')
    #model = torch.compile(model)

    new_checkpoint = dict()
    for key, val in checkpoint['model'].items():

        new_checkpoint[key.replace('_orig_mod.', '')] = val

    model.load_state_dict(new_checkpoint)

    aggr = Aggregator().to('cuda')
    #aggr = torch.compile(aggr)
    new_checkpoint = dict()

    for key, val in checkpoint['aggr'].items():

        new_checkpoint[key.replace('_orig_mod.', '')] = val
            
    aggr.load_state_dict(new_checkpoint)

    model_ema = EMA(model, decay = 0.999)
    aggr_ema = EMA(aggr, decay = 0.999)

    new_checkpoint = dict()
    for key, val in checkpoint['model_ema'].items():

        new_checkpoint[key.replace('_orig_mod.', '')] = val
    model_ema.load_state_dict(new_checkpoint)

    new_checkpoint = dict()
    for key, val in checkpoint['aggr_ema'].items():

        new_checkpoint[key.replace('_orig_mod.', '')] = val
    aggr_ema.load_state_dict(new_checkpoint)

    preprocess = Preprocess()

    logit_dict = dict()

    for idx, test_file_name in enumerate(test_file_names):
        
        print('[info] {}/{}'.format(idx, len(test_file_names)))
        # face video 만들기
        video_path = os.path.join(args['data_test_path'], test_file_name)
        face_video_path = os.path.join(args['data_path'], 'test', test_file_name)

        #os.makedirs(face_video_path, exist_ok=True)

        success = preprocess.make_face_video(src_video_path=video_path, dst_video_path=face_video_path)

        #assert success is True, test_file_name

        if not success:

            submission.loc[submission['path'] == test_file_name, 'label'] = 'real'
            continue

        #inference 하기
        video = make_video_tensor(face_video_path, 16)
        fl, _, w, h = video.size()
        video = video.view(fl, 3, w, h).to('cuda')
        
        model.eval()
        aggr.eval()

        if use_ema:
            model_ema.apply_shadow()
            aggr_ema.apply_shadow()
        
        with torch.no_grad():
            emb = model(video) # (bs * fl, dim, w, h)
            bsfl, d, w, h = emb.size()
            emb = emb.unsqueeze(0)

            logit = aggr(emb)
            prob = torch.sigmoid(logit)
            pred = (prob > threshold).float()

            logit_dict[test_file_name] = logit.item()

        submission.loc[submission['path'] == test_file_name, 'label'] = 'fake' if pred == 1.0 else 'real'

    submission.to_csv('/home/elicer/sample_submission_{}.csv'.format(name), index=False)

    with open('/home/elicer/logit_{}.pkl'.format(name), 'wb') as fs:

        pickle.dump(logit_dict, fs)


# file 있으면 건너뛰기

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type = str,
                        help = 'config file')
    parser.add_argument('--filename', type = str,
                        help = 'checkpoint file name')
    parser.add_argument('--omp-num-threads', type = int,
                        default = 2,
                        help = 'OMP_NUM_THREADS option')
    parser.add_argument('--ema', action='store_true',
                        help = 'use ema')
    parser.add_argument('--name', type = str,
                        help = 'submission name')

    opt = vars(parser.parse_args())

    with codecs.open(opt['conf'], 'r', encoding = 'UTF-8') as fs: exec(fs.read())

    os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args, checkpoint_file_name = opt['filename'], use_ema=opt['ema'], name = opt['name'])
