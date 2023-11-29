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
        output_face_path = os.path.join(output_folder, folder_name, 'face')
        output_numpy_path = os.path.join(output_folder, folder_name, 'numpy')

        # output 폴더 생성
        if not os.path.exists(output_face_path):
            os.makedirs(output_face_path)
        
        # output 폴더 생성
        if not os.path.exists(output_numpy_path):
            os.makedirs(output_numpy_path)

        video_files = glob(os.path.join(folder_path, '*.mp4'))
        video_files = sorted(video_files, key=lambda x: x)

        # 각 비디오 파일에 대해 처리
        for idx, video_file in enumerate(video_files):
            filename = os.path.basename(video_file)
            output_face_file = os.path.join(output_face_path, filename)
            output_numpy_file = os.path.join(output_numpy_path, filename.replace('.mp4', '.npy'))

            preprocess.make_face_video(src_video_path=video_file, dst_video_path=output_face_file, dst_numpy_path=output_numpy_file)
            preprocess.print_log(f'{idx+1}/{len(video_files)} 작업 완료')

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
        preprocess.print_log(f'{idx+1}/{len(video_files)} 작업 완료')
            
def process_videos(input_path: str, output_path: str, dataset: str) -> None:
    preprocess = Preprocess()
    if dataset=='train':
        process_folder(input_path, output_path, preprocess)
    elif dataset=='test':
        process_test(input_path, output_path, preprocess)

def main(args: Dict[str, Any]) -> None:
    process_videos(args['in_path'], args['out_path'], args['dataset'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type = str,
                        help = 'input data path')
    parser.add_argument('--out-path', type = str,
                        help = 'output data path')
    parser.add_argument('--omp-num-threads', type = int,
                        default = 2,
                        help = 'OMP_NUM_THREADS option')
    parser.add_argument('--dataset', type = str,
                        help = 'choose train test')
                        
    args = vars(parser.parse_args())

    # os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args)
