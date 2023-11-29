import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, Iterator, Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import Preprocess

def main(args: Dict[str, Any]) -> None:
    preprocess = Preprocess()
    preprocess.make_face_video()

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

    # os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args)
