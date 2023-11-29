import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, Iterator, Iterable
import pandas as pd
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import Preprocess

def process_folder(input_folder: str, output_folder: str, preprocess: Preprocess) -> None:
    # 입력 폴더 내의 모든 서브폴더에 대한 처리
    for folder_name in ['fake', 'real']:
        folder_path = os.path.join(input_folder, folder_name)
        output_path = os.path.join(output_folder, folder_name)

        # output 폴더 생성
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        video_files = glob(os.path.join(folder_path, '*.mp4'))  

        # 각 비디오 파일에 대해 처리
        for video_file in video_files:
            filename = os.path.basename(video_file)
            output_file = os.path.join(output_path, filename)

            print(f"Processing {video_file}...")
            preprocess.make_face_video(src_video_path=video_file, dst_video_path=output_file)
            print(f"Finished processing {video_file}, saved to {output_file}")

def process_videos(input_path: str, output_path: str) -> None:
    preprocess = Preprocess()
    process_folder(input_path, output_path, preprocess)

def main(args: Dict[str, Any]) -> None:
    process_videos(args['in_path'], args['out_path'])

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
