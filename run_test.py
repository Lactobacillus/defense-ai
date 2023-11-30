import os
import sys
import codecs
import argparse
import torch
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from model.model import CustomResNet50, Aggregator, LinearLayer
from data.dataset import VideoStage1Data, VideoPretrainData

def make_video_numpy(video_path, frame_length) -> np.ndarray:
    fn = video_path
    video = video2numpy(fn)

    start = random.randrange(0, video.size(0) - frame_length - 1)
    end = start + frame_length
    cut = np.transpose(video[start:end, ...], (0, 3, 1, 2))

    return cut

def video2numpy(filepath: str) -> np.ndarray:

    cap = cv2.VideoCapture(filepath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    return buf

def main(args: Dict[str, Any],
        checkpoint_file_name: int,
        use_ema:bool,
        threshold: float = 0.5) -> None:

    checkpoint_file_name = checkpoint_file_name.replace('.pkl', '')
    checkpoint_file_path = os.path.join(args['checkpoints_path'], f'{checkpoint_file_name}.pkl')
    checkpoint = torch.load(checkpoint_file_path)

    submission = pd.read_csv('/home/elicer/sample_submission.csv')
    test_file_names = submission['path'].tolist()

    preprocess = Preprocess()

    for test_file_name in test_file_names:
        # face video 만들기
        video_path = os.path.join(args['data_test_path'], test_file_name)
        face_video_path = os.path.join(args['data_path'], 'test', test_file_name)

        os.makedirs(os.path.join(args['data_path'], 'test'), exist_ok=True)

        success = preprocess.make_face_video(src_video_path=video_path, dst_video_path=face_video_path)

        if not success:
            submission.loc[submission['path'] == test_file_name, 'label'] = 'real'
            continue

        # 모델 불러오기
        processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
        model = CustomResNet50(pool=True).to('cuda')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        linear = LinearLayer().to('cuda')
        linear.load_state_dict(checkpoint['linear'])
        linear.eval()
        # new_checkpoint = dict()
        # for key, val in checkpoint['model'].items():

        #     new_checkpoint[key.replace('_orig_mod.', '')] = val

        # model.load_state_dict(new_checkpoint)

        #inference 하기
        with torch.no_grad():
            video = make_video_numpy(face_video_path, 16).to('cuda')
            print(video.size())
            fl, _, w, h = video.size()
            video = video.view(fl, 3, w, h)

            pixel = processor(video, return_tensors = 'pt').pixel_values.to('cuda')
            emb = model(pixel) # (bs * fl, dim, w, h)

            logit = linear(emb)
            prob = torch.sigmoid(logit)
            pred = (prob > threshold).float()

        submission.loc[submission['path'] == test_file_name, 'label'] = 'fake' if pred == 0.0 else 'real'

    submission.to_csv('/home/elicer/sample_submission_test.csv', index=False)


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
