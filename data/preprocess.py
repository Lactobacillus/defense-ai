from typing import List, Tuple, Optional
import os
import cv2
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


class Preprocess:
    def __init__(self) -> None:
        self.src_video_path: str = None
        self.dst_video_path: str = None
        self.output_size: Tuple[int, int] = None

        self.insight_face_app = FaceAnalysis(name='buffalo_sc')
        self.insight_face_app.prepare(ctx_id=0, det_size=(480,480))
    
    def print_log(self, text) -> None:
        """
            print log with timestamp(mm-dd HH:MM:SS)  
        """
        print(f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")}][LOG] {text}')

    def video2numpy(self, extract_frame_num) -> List[np.ndarray]:
        cap = cv2.VideoCapture(self.src_video_path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        buf = []

        fc = 0
        ret = True

        while (fc < frameCount and ret):
            
            # ret, buf[fc] = cap.read()
            ret, frame = cap.read()
            
            buf.append(frame)
            fc += 1
            
            if fc >= extract_frame_num * 2:
                break

        cap.release()

        return buf

    def extract_faces(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        원본 frame list에서 얼굴 부분만 추출하여 list 형태로 반환합니다.
        첫 번째 프레임에서 탐지된 바운딩 박스를 사용하고, 모든 프레임에 동일하게 적용합니다.
        """
        fixed_bbox = None
        face_list=[]
        for frame in frames:
            if fixed_bbox is None:
                faces = self.insight_face_app.get(frame)
                if len(faces) != 1:
                    self.print_log('얼굴 탐지 실패 시 다음 프레임으로 넘어감')
                    continue

                bbox = faces[0].bbox.astype(int)
                # 10% 마진 추가
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                margin_w, margin_h = width * 0.05, height * 0.05
                fixed_bbox = [max(bbox[0] - margin_w, 0), max(bbox[1] - margin_h, 0),
                            min(bbox[2] + margin_w, frame.shape[1]), min(bbox[3] + margin_h, frame.shape[0])]
            
            cropped_face = frame[int(fixed_bbox[1]):int(fixed_bbox[3]), int(fixed_bbox[0]):int(fixed_bbox[2])]
            
            # 원본 이미지의 높이와 너비
            h, w = cropped_face.shape[:2]

            # 배경이 검은 이미지의 너비와 높이 (128x128)
            new_width = max(h, w)
            new_height = max(h, w)

            # 새 이미지에 대한 검은색 배경 생성
            new_img = np.zeros((new_height, new_width, 3), np.uint8)

            # 원본 이미지를 가운데에 배치하기 위한 시작점 계산
            start_x = (new_width - w) // 2
            start_y = (new_height - h) // 2

            # 원본 이미지를 새 이미지(검은색 배경)에 복사
            new_img[start_y:start_y+h, start_x:start_x+w] = cropped_face

            resized_face = cv2.resize(new_img, self.output_size, interpolation=cv2.INTER_AREA)

            face_list.append(resized_face)
        
        return face_list

    def save_face_video(self, faces: List[np.ndarray], save_video_path: str = None, save_numpy_path: str = None) -> bool:
        try:
            assert save_video_path != None and save_numpy_path != None

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_video_path, fourcc, 30.0, self.output_size)

            # 비디오에 프레임 추가
            for face in faces:
                out.write(face)
            
            out.release()

            numpy_faces = np.array(faces)
            np.save(save_numpy_path, numpy_faces)

            return True

        except Exception as e:
            print(f'[ERROR] Fail : save_face_video, reason : {e}')
            
            return False


    def make_face_video(self,
                src_video_path: str = 'C:/Users/Admin/Downloads/참가자용 데이터(학습데이터,설치파일,발표자료)/Train Dataset_비디오 분야 과제/train/real/89f7ad6b7ae6481fac5d3e8992d22127.mp4',
                dst_video_path: str = './output.mp4',
                dst_numpy_path: str = './output.npy',
                width: int = 128,
                height: int = 128,
                extract_frame_num = 64
        ) -> bool:
        """
            src_video_path : 원본 영상 path
            dst_video_path : face output 영상 path
            width : output video 가로
            height : output video 세로
        """
        self.src_video_path = src_video_path
        self.dst_video_path = dst_video_path
        self.output_size = (width, height)

        frames = self.video2numpy(extract_frame_num=extract_frame_num)
        self.print_log(f'Original video shape : [{len(frames)}, {frames[0].shape}]')
        faces = self.extract_faces(frames[::2])
        
        if len(faces) != extract_frame_num:
            self.print_log(len(faces))
            return False
        
        self.print_log(f'Faces video shape : [{len(faces)}, {faces[0].shape}]')

        success = self.save_face_video(faces=faces, save_video_path=self.dst_video_path, save_numpy_path=dst_numpy_path)

        if success:
            self.print_log(f'성공적으로 얼굴 비디오를 생성했습니다. 저장 위치 : {self.dst_video_path}')
        else:
            self.print_log('얼굴 비디오 생성에 실패하였습니다.')

        return success

    def RGB_mean_var(folder_path, frame_index=0):
            
            mean_squared_len = []
            mean_len = []
            total_length = 0
            file_list = os.listdir(folder_path)
            
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                cap = cv2.VideoCapture(file_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                cap.release()
                
                m = frame.mean(axis=(0, 1))
                mean_squared_len.append(m**2*length)
                mean_len.append(m*length)
                total_length += length
                
            MEAN = np.array(mean_len).sum(axis=0) / total_length
            VAR = (np.array(mean_squared_len).sum(axis=0) / total_length) - (MEAN**2)
            
            
            return MEAN, VAR
