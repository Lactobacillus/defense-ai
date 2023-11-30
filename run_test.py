import os
import sys
import codecs
import argparse
import torch
import numpy as np
import pandas as pd
import random
import cv2
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
import glob

from transformers import AutoImageProcessor

from model.model import CustomResNet50, Aggregator, LinearLayer
from data.dataset import VideoStage1Data, VideoPretrainData, TestDataset

from data.preprocess import Preprocess

def make_video_numpy(video_path, frame_length) -> np.ndarray:
    fn = video_path
    video = video2numpy(fn)

    start = random.randrange(0, video.shape[0] - frame_length + 1)
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

def process_test(input_folder: str, output_folder: str, preprocess: Preprocess) -> None:
    # output 폴더 생성
    output_face_path = os.path.join(output_folder, 'face')
    output_numpy_path = os.path.join(output_folder, 'numpy')

    if not os.path.exists(output_face_path):
        os.makedirs(output_face_path)
    
    if not os.path.exists(output_numpy_path):
        os.makedirs(output_numpy_path)

    video_files = glob.glob(os.path.join(input_folder, '*.mp4'))
    video_files = sorted(video_files, key=lambda x: x)

    # 각 비디오 파일에 대해 처리
    for idx, video_file in enumerate(video_files):
        filename = os.path.basename(video_file)
        output_face_file = os.path.join(output_face_path, filename)
        output_numpy_file = os.path.join(output_numpy_path, filename.replace('.mp4', '.npy'))

        preprocess.make_face_video(src_video_path=video_file, dst_video_path=output_face_file, dst_numpy_path=output_numpy_file)
        preprocess.print_log(f'{idx+1}/{len(video_files)} 영상 처리 작업 완료')


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

    # face video 만들기
    process_test(input_folder=args['data_test_path'], output_folder=os.path.join(args['data_path'], 'test'), preprocess=preprocess)
    
    dset = TestDataset(data_path=os.path.join(args['data_path'], 'test/face'), frame_length=args['frame_length'])
    train_loader = DataLoader(dataset = dset,
            batch_size = args['batch_size'],
            shuffle = False,
            drop_last = False,
            num_workers = 4,
            pin_memory = True)
    
    # 모델 불러오기
    processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
    model = CustomResNet50(pool=True).to('cuda')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    linear = LinearLayer().to('cuda')
    linear.load_state_dict(checkpoint['linear'])
    linear.eval()

    for idx, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
        #inference 하기
        with torch.no_grad():
            video = batch['video'].to('cuda')
            files = batch['file_name']
            print(len(files), files)
            bs, fl, _, w, h = video.size()
            video = video.view(bs * fl, 3, w, h)
            pixel = processor(video, return_tensors = 'pt').pixel_values.to('cuda')
            emb = model(pixel) # (bs * fl, dim, w, h)

            logit = linear(emb)
            prob = torch.sigmoid(logit)
            prob = prob.view(bs, fl, 1)
            pred = (prob > threshold).float()
            mean_prob = torch.mean(pred, dim=1) # torch.size([4, 1])

        for idx, fn in enumerate(files):
            submission.loc[submission['path'] == fn, 'label'] = 'fake' if mean_prob[idx, 0] < 0.5 else 'real'

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
