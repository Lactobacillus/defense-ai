import os
import sys
import cv2
import random
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, Iterator, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader


class VideoPretrainData(Dataset):

    def __init__(self,
            data_path: str,
            frame_length: int = 16) -> None:

        # assumption: using only real data
        self.data_path = data_path
        self.frame_length = frame_length
        self.fn_list = os.listdir(data_path)

    def __len__(self) -> int:

        return len(self.fn_list)

    def __getitem__(self,
            idx: int) -> Dict[str, Any]:

        fn = self.fn_list[idx]
        vid = self.video2numpy(os.path.join(self.data_path, fn))

        start = random.randrange(0, vid.shape[0] - self.frame_length - 1)
        end = start + self.frame_length
        cut = np.transpose(vid[start:end, ...], (0, 3, 1, 2))

        return {'vid': cut}

    def video2numpy(self,
            filepath: str) -> np.ndarray:

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


class VideoStage1Data(Dataset):

    def __init__(self,
            data_path: str,
            frame_length: int = 16,
            split: str = 'train') -> None:

        self.data_path = data_path
        self.frame_length = frame_length

        real_list = os.listdir(os.path.join(data_path, split, 'real'))
        fake_list = os.listdir(os.path.join(data_path, split, 'fake'))
        self.pair = [(os.path.join(data_path, split, 'real', fn), 1) for fn in real_list]
        self.pair = self.pair + [(os.path.join(data_path, split, 'fake', fn), 0) for fn in fake_list]

    def __len__(self) -> int:

        return len(self.pair)

    # def __getitem__(self,
    #         idx: int) -> Dict[str, Any]:

    #     fn, label = self.pair[idx]
    #     vid = self.video2numpy(fn)

    #     start = random.randrange(0, vid.shape[0] - self.frame_length - 1)
    #     end = start + self.frame_length
    #     cut = np.transpose(vid[start:end, ...], (0, 3, 1, 2))

    #     return {'video': cut, 'label': label}

    def __getitem__(self,
            idx: int) -> Dict[str, Any]:

        fn, label = self.pair(idx)
        video = self.video2tensor(fn)

        return {'video': video, 'label': label}

    def video2numpy(self,
            filepath: str) -> np.ndarray:

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

    def video2tensor(self,
            filename: str) -> torch.Tensor:

        video, _, _ = io.read_video(filename)

        return video
