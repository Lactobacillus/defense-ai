import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, Iterator, Iterable


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


def main(args: Dict[str, Any]) -> None:

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type = str,
                        help = 'input data path')
    parser.add_argument('--out-path', type = str,
                        help = 'output data path')
    parser.add_argument('--omp-num-threads', type = int,
                        default = 2,
                        help = 'OMP_NUM_THREADS option')
    args = vars(parser.parse_args())

    os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args)
