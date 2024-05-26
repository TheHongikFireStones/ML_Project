import os
import dlib
import cv2

#google colab 사용시 넣어야 할 코드
from google.colab import drive
drive.mount('/content/drive')

# 얼굴 탐지기 로드
detector = dlib.get_frontal_face_detector()

# 이미지가 저장된 디렉토리 경로
image_dir = '/content/drive/MyDrive/ML_Project_/'
actors_list = ["Aaron Diaz", "Alexandre Cunha", "Bernardo Velasco", "Diego Boneta", "Francisco Lachowski", "Henry Zaga", "Leonardo Sbaraglia", "Marlon Teixeira", "Oscar Isaac", "Santiago Cabrera"]

def delete_non_face_images(directory):
    # 디렉토리 내의 모든 파일을 순회
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)

            # 이미지 로드
            image = cv2.imread(file_path)
            if image is None:
                os.remove(file_path) # 파일 삭제
                print(f"이미지 파일을 열 수 없습니다: {filename}")
                continue

            # dlib은 RGB 이미지를 사용하므로 BGR을 RGB로 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 얼굴 탐지
            faces = detector(rgb_image)

            # 얼굴이 없는 경우 파일 삭제
            if len(faces) == 0:
                os.remove(file_path)
                print(f"얼굴이 없어 파일을 삭제했습니다: {filename}")


for actor in actors_list:
    path = image_dir + actor
    print(path)
    delete_non_face_images(path)