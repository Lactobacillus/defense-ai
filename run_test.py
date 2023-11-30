import os
import sys
import codecs
import argparse
import torch
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

from data.preprocess import Preprocess
from model.model import CustomResNet50, Aggregator
from data.dataset import VideoStage1Data
from train.util import LossAccumulator, EMA

def make_video_tensor(video_path, frame_length) -> torch.Tensor:
    fn = video_path
    video = video2tensor(fn)

    start = random.randrange(0, video.size(0) - self.frame_length - 1)
    end = start + self.frame_length
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
        threshold: float = 0.5) -> None:

    checkpoint_file_name = checkpoint_file_name.replace('.pkl', '')
    checkpoint_file_path = os.path.join(args['result_path'], f'{checkpoint_file_name}.pkl')
    checkpoint = torch.load(checkpoint_file_path)

    submission = pd.read_csv('sample_submission.csv')
    test_file_names = submission['path'].tolist()

    preprocess = Preprocess()

    for test_file_name in test_file_names:
        # face video 만들기
        video_path = os.path.join(self.args['data_test_path'], test_file_name)
        face_video_path = os.path.join(self.args['data_path'], 'test', test_file_name)

        os.makedirs(face_video_path, is_exists=True)

        suceess = preprocess.make_face_video(src_video_path=video_path, dst_video_path=face_video_path)

        assert success is True

        # 모델 불러오기
        model = CustomResNet50().to('cuda')
        model = torch.compile(model)
        model.load_state_dict(checkpoint['model'])

        aggr = Aggregator().to('cuda')
        aggr = torch.compile(aggr)
        aggr.load_state_dict(checkpoint['aggr'])

        model_ema = EMA(model, decay = 0.999)
        aggr_ema = EMA(aggr, decay = 0.999)
        model_ema.load_state_dict(checkpoint['model_ema'])
        aggr_ema.load_state_dict(checkpoint['aggr_ema'])

        #inference 하기
        video = make_video_tensor(face_video_path)
        bs, fl, _, w, h = video.size()
        video = video.view(bs * fl, 3, w, h)
        
        model.eval()
        aggr.eval()
        model_ema.eval()
        aggr_ema.eval()

        if use_ema:
            model_ema.apply_shadow()
            aggr_ema.apply_shadow()
        
        with torch.no_grad():
            emb = model(video) # (bs * fl, dim, w, h)
            bsfl, d, w, h = emb.size()
            emb = emb.view(bs, fl, d, w, h)

            logit = aggr(emb)
            prob = torch.sigmoid(logit)
            pred = (prob > threshold).float()

        df.loc[df['path'] == test_file_name, 'label'] = 'real' if pred == 0.0 else 'fake'

    submission.to_csv('sample_submission_test.csv', index=False)


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

    opt = vars(parser.parse_args())

    with codecs.open(opt['conf'], 'r', encoding = 'UTF-8') as fs: exec(fs.read())

    os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args, checkpoint_file_name = opt['filename'], use_ema=opt['ema'])
