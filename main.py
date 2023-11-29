import pandas as pd
from glob import glob
import cv2
from matplotlib import pyplot as plt


TRAIN_PATH = 'C:/Users/Admin/Downloads/참가자용 데이터(학습데이터,설치파일,발표자료)/Train Dataset_비디오 분야 과제/train'
# TEST_PATH = '/mnt/elice/dataset/test'
train_files = sorted(glob(TRAIN_PATH+'/*/*'))
print(len(train_files))

labels = [label.split('/')[-2] for label in train_files]
train = pd.DataFrame({'path':train_files, 'label':labels})

def video_to_image(src: str) -> str:
    stream = cv2.VideoCapture(src)

    (grabbed, frame) = stream.read()
    img_cv2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    plt.imshow(img_cv2)
    plt.show()

print(train['path'][0])
video_to_image(train['path'][0])
video_to_image(train['path'][1])

# submission = pd.read_csv('sample_submission.csv')
# submission['label'] = 'fake'
# submission.to_csv('sample_submissison.csv', index=False)