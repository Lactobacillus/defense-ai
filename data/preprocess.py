from typing import List
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from matplotlib import pyplot as plt

class Preprocess:
    def __init__(self,
                 video_path: str = 'C:/Users/Admin/Downloads/참가자용 데이터(학습데이터,설치파일,발표자료)/Train Dataset_비디오 분야 과제/train/real/0a13784b1e3841188b923b41a673b2e6.mp4'
        ) -> None:
        self.video_path: str = video_path
        self.insight_face_app = FaceAnalysis()
        self.insight_face_app.prepare(ctx_id=0, det_size=(320,320))

    def video2numpy(self) -> List:
        cap = cv2.VideoCapture(self.video_path)
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

    def extract_faces(self, frames: List):
        for frame in frames:
            faces = self.insight_face_app.get(frame)
            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            img_cv2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            plt.imshow(img_cv2)
            plt.show()
        
        cv2.destroyAllWindows()

    def make_face_video(self, dst_video_path):
        print(self.video_path)
        frames = self.video2numpy()

        self.extract_faces(frames)
        print(frames.shape)
