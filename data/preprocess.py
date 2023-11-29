from typing import List, Tuple
import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from matplotlib import pyplot as plt
import datetime

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
            원본 frame list 에서 얼굴 부분만 추출하여 list 형태로 반환합니다. 
        """
        face_list = []
        for frame in frames:
            faces = self.insight_face_app.get(frame)

            if len(faces) != 1:
                self.print_log(f'Face 개수가 이상함 : {len(faces)}')

            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                cropped_face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

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
                src_video_path: str = 'C:/Users/Admin/Downloads/참가자용 데이터(학습데이터,설치파일,발표자료)/Train Dataset_비디오 분야 과제/train/real/0a13784b1e3841188b923b41a673b2e6.mp4',
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
            return False
        
        self.print_log(f'Faces video shape : [{len(faces)}, {faces[0].shape}]')

        success = self.save_face_video(faces=faces, save_video_path=self.dst_video_path, save_numpy_path=dst_numpy_path)

        if success:
            self.print_log(f'성공적으로 얼굴 비디오를 생성했습니다. 저장 위치 : {self.dst_video_path}')
        else:
            self.print_log('얼굴 비디오 생성에 실패하였습니다.')

        return success

    def make_face_video_all(self,
            src_video_folder_path: str = 'C:/Users/Admin/Downloads/참가자용 데이터(학습데이터,설치파일,발표자료)/Train Dataset_비디오 분야 과제/train/fake/',
            dst_video_folder_path: str = './outputs/fake',
            width: int = 128,
            height: int = 128
        ) -> bool:
        """
            src_video_path : 원본 영상 폴더 path
            dst_video_path : face video(output) 폴더 path
            width : output video 가로
            height : output video 세로
        """
        face_video_folder_path = os.path.join(dst_video_folder_path, 'face')
        face_numpy_folder_path = os.path.join(dst_video_folder_path, 'numpy')
        file_names = sorted(os.listdir(src_video_folder_path), key=lambda x: x)
        
        if not os.path.exists(face_video_folder_path):
            os.makedirs(face_video_folder_path)

        if not os.path.exists(face_numpy_folder_path):
            os.makedirs(face_numpy_folder_path)
        
        self.print_log(f'{src_video_folder_path} 작업 시작')
        for idx, file_name in enumerate(file_names):
            if idx <1642:
                continue
            src_path = os.path.join(src_video_folder_path, file_name)
            dst_path = os.path.join(face_video_folder_path, file_name)
            numpy_path = os.path.join(face_numpy_folder_path, file_name.replace('.mp4', '.npy'))

            success = self.make_face_video(src_video_path=src_path, dst_video_path=dst_path, dst_numpy_path=numpy_path, width=width, height=height)
            if success:
                self.print_log(f'성공적으로 얼굴 비디오를 생성했습니다. 저장 위치 : {self.dst_video_path}')
            else:
                self.print_log('얼굴 비디오 생성에 실패하였습니다.')

            self.print_log(f'{idx+1}/{len(file_names)} 번째 영상 작업 완료')
