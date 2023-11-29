from typing import List
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from matplotlib import pyplot as plt

class Preprocess:
    def __init__(self) -> None:
        self.src_video_path: str = None
        self.dst_video_path: str = None
        self.output_size: tuple(int, int) = None

        self.insight_face_app = FaceAnalysis()
        self.insight_face_app.prepare(ctx_id=0, det_size=(480,480))
        

    def video2numpy(self) -> List[np.ndarray]:
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

        cap.release()

        return buf

    def extract_faces(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
            원본 frame list 에서 얼굴 부분만 추출하여 list 형태로 반환합니다. 
        """
        face_list = []
        for frame in frames:
            faces = self.insight_face_app.get(frame)

            assert len(faces) == 1

            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                cropped_face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                resized_face = cv2.resize(cropped_face, self.output_size, interpolation=cv2.INTER_AREA)

                face_list.append(resized_face)
        
        return face_list

    def save_face_video(self, faces: List[np.ndarray]):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.dst_video_path, fourcc, 30.0, self.output_size)

            # 비디오에 프레임 추가
            for face in faces:
                out.write(face)
            
            out.release()

            return True

        except Exception as e:
            print(f'[ERROR] Fail : save_face_video, reason : {e}')
            
            return False


    def make_face_video(self,
                src_video_path: str = 'C:/Users/Admin/Downloads/참가자용 데이터(학습데이터,설치파일,발표자료)/Train Dataset_비디오 분야 과제/train/real/0a13784b1e3841188b923b41a673b2e6.mp4',
                dst_video_path: str = './output.mp4',
                width: int = 256,
                height: int = 256
        ):
        """
            src_video_path : 원본 영상 path
            dst_video_path : face output 영상 path
            width : output video 가로
            height : output video 세로
        """
        self.src_video_path = src_video_path
        self.dst_video_path = dst_video_path
        self.output_size = (width, height)

        frames = self.video2numpy()
        print(f'[LOG] Original video shape : [{len(frames)}, {frames[0].shape}]')
        faces = self.extract_faces(frames[:100])
        print(f'[LOG] Faces video shape : [{len(faces)}, {faces[0].shape}]')
        success = self.save_face_video(faces)
        if success:
            print(f'[LOG] 성공적으로 얼굴 비디오를 생성했습니다. 저장 위치 : {self.dst_video_path}')
        else:
            print('[LOG] 얼굴 비디오 생성에 실패하였습니다.')
