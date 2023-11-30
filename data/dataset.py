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
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


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

        start = random.randrange(0, vid.shape[0] - self.frame_length + 1)
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

        fn, label = self.pair[idx]
        video = self.video2numpy(fn)

        start = random.randrange(0, video.shape[0] - self.frame_length + 1)
        end = start + self.frame_length
        # video = video[start:end, ...]
        cut = np.transpose(video[start:end, ...], (0, 3, 1, 2))

        return {'video': cut, 'label': label}

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
            filename: str,
            output_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:

        video, _, _ = tv.io.read_video(filename, output_format = 'TCHW')

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(output_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])

        video = torch.stack([transform(frame) for frame in video])

        return video


# 영상 읽는 것을 배치로
class TestDataset(Dataset):

    def __init__(self,
            data_path: str,
            frame_length: int = 16,
            split: str = 'train') -> None:

        self.data_path = data_path
        self.frame_length = frame_length

        file_name_list = os.listdir(data_path)
        self.pair = [(os.path.join(data_path, filename), filename) for filename in file_name_list]

    def __len__(self) -> int:

        return len(self.test_file_list)

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
        fn, filename = self.pair[idx]
        video = self.video2numpy(fn)

        start = random.randrange(0, video.shape[0] - self.frame_length + 1)
        end = start + self.frame_length
        # video = video[start:end, ...]
        cut = np.transpose(video[start:end, ...], (0, 3, 1, 2))

        return {'video': cut, 'file_name': filename}

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
            filename: str,
            output_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:

        video, _, _ = tv.io.read_video(filename, output_format = 'TCHW')

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(output_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])

        video = torch.stack([transform(frame) for frame in video])

        return video
