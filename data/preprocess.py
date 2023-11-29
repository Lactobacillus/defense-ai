from typing import List
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from matplotlib import pyplot as plt
from typing import List, Optional
from insightface.utils.face_align import norm_crop

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
    
    def align_face(self, face: np.ndarray, src_landmarks: np.ndarray, target_landmarks: np.ndarray) -> np.ndarray:
        """
        얼굴을 정렬하는 함수입니다. 
        src_landmarks는 현재 프레임의 얼굴 랜드마크이고,
        target_landmarks는 정렬 기준이 될 랜드마크입니다.
        """
        # 현재 얼굴의 랜드마크 조정
        src = src_landmarks.astype(np.float32)

        # 타겟 랜드마크 조정
        target = target_landmarks.astype(np.float32)

        # 변환 매트릭스 계산
        transform_matrix = cv2.estimateAffinePartial2D(src, target)[0]

        # 얼굴 정렬
        aligned_face = cv2.warpAffine(face, transform_matrix, self.output_size, borderValue=0.0)

        return aligned_face
    
    def find_first_valid_landmarks(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        첫 번째 유효한 랜드마크를 가진 프레임을 찾는 함수입니다.
        유효한 랜드마크를 찾지 못하면 None을 반환합니다.
                        bbox = face.bbox.astype(int)
                landmarks = face.kps.astype(int)
                # 바운딩 박스에 맞춰 랜드마크 조정
                adjusted_landmarks = landmarks.copy()
                adjusted_landmarks[:, 0] -= bbox[0]
                adjusted_landmarks[:, 1] -= bbox[1]
        """
        for frame in frames:
            faces = self.insight_face_app.get(frame)
            for face in faces:
                if face is not None and face.kps is not None:
                    print("랜드마크를 찾았습니다.")
                    return face.kps
                
        print("랜드마크를 찾지 못했습니다.")
        return None
        
        

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
                    continue  # 얼굴 탐지 실패 시 다음 프레임으로 넘어감

                bbox = faces[0].bbox.astype(int)
                # 10% 마진 추가
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                margin_w, margin_h = width * 0.05, height * 0.05
                fixed_bbox = [max(bbox[0] - margin_w, 0), max(bbox[1] - margin_h, 0),
                            min(bbox[2] + margin_w, frame.shape[1]), min(bbox[3] + margin_h, frame.shape[0])]
            
            cropped_face = frame[int(fixed_bbox[1]):int(fixed_bbox[3]), int(fixed_bbox[0]):int(fixed_bbox[2])]
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
                src_video_path: str = 'C:/Users/Admin/Downloads/참가자용 데이터(학습데이터,설치파일,발표자료)/Train Dataset_비디오 분야 과제/train/real/89f7ad6b7ae6481fac5d3e8992d22127.mp4',
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