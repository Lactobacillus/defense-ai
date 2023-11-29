import subprocess
import sys

# requirements.txt 파일 읽기
with open('requirements.txt', 'r') as file:
    requirements = file.readlines()

# 각 패키지를 개별적으로 설치
for req in requirements:
    req = req.strip()
    if req:  
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError:
            print(f"Failed to install {req}, skipping...")
