import os
import numpy as np

import cv2
import dlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

MOUTH_LEFT_IDX = 48
MOUTH_RIGHT_IDX = 54


# 입의 크기를 구하는 함수
def get_mouth_width(face_left, face_right, landmarks):
    return (landmarks.part(MOUTH_RIGHT_IDX).x - landmarks.part(MOUTH_LEFT_IDX).x) / (face_right - face_left)


# 얼굴의 각도를 구하는 함수
def get_face_slope_pair(landmarks):
    try:
        left_top_slope = abs(landmarks.part(0).y - landmarks.part(4).y) / abs(landmarks.part(0).x - landmarks.part(4).x)
        right_top_slope = abs(landmarks.part(16).y - landmarks.part(12).y) / abs(
            landmarks.part(16).x - landmarks.part(12).x)

        left_bottom_slope = abs(landmarks.part(5).y - landmarks.part(8).y) / abs(
            landmarks.part(5).x - landmarks.part(8).x)
        right_bottom_slope = abs(landmarks.part(11).y - landmarks.part(8).y) / abs(
            landmarks.part(11).x - landmarks.part(8).x)

        top_slope_avg = (left_top_slope + right_top_slope) / 2
        bottom_slope_avg = (left_bottom_slope + right_bottom_slope) / 2
        return top_slope_avg, bottom_slope_avg
    except ZeroDivisionError:
        return 0, 0


def visualize_first_feature():
    first_feature_values = [item[0] for item in features]
    # Create jittered x values
    x_values = np.random.normal(1, 0.01, len(first_feature_values))
    plt.scatter([1] * len(first_feature_values), first_feature_values, s=10)
    # Label the axes
    plt.xticks([1], ['Feature 1'])
    plt.xlabel('Feature 1')
    plt.ylabel('Value')
    # Display the plot
    plt.show()


# 이미지 경로
image_dir = './img/'
# 디렉토리 이름 리스트
actors_list = ["Aditya_Roy_Kapur", "Arjun_Rampal", "Hrithik_Roshan", "John_Abraham", "Kartik_Aaryan", "Ranveer_Singh",
               "Shahid_Kapoor", "Sidharth_Malhotra", "Sidharth_Malhotra", "Varun_Dhawan"]

# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 데이터 셋을 저장할 리스트
# data = [
#    [0.1, 0.2, 0.3, (0.4, 0.5), 'Race1'],
#    [0.2, 0.3, 0.4, (0.5, 0.6), 'Race2'],
#    [0.3, 0.4, 0.5, (0.6, 0.7), 'Race3'],
#    [0.4, 0.5, 0.6, (0.7, 0.8), 'Race4'],
#    ]
# 위와 같은 형식으로 데이터를 저장.
# 단, 마지막 요소는 label이다.
data_set = []

# 각 배우들의 사진을 담은 디렉토리들을 순회
for actor in actors_list:
    path = image_dir + actor
    # 디렉토리 내에 있는 사진들을 순회
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        # 이미지 로드
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = detector(gray)

        # 얼굴이 여러 개라면 각 얼굴들에 대해 작업 수행
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            # 랜드마크 검출
            landmarks = predictor(gray, face)

            # 각 얼굴들에 대해 feature를 추출한 후 데이터 셋에 넣는다.
            data_set.append([
                # 입 크기
                get_mouth_width(x1, x2, landmarks),
                # 얼굴이 각진 정도
                get_face_slope_pair(landmarks),
                # 인종 (label)
                'Indian'
            ])

# feature와 label을 각각 분리해 담을 리스트
features = []
labels = []

# feature와 label을 분리한다.
for item in data_set:
    feature_vector = item[:1] + list(item[1])  # Flatten the tuple into two separate features
    features.append(feature_vector)
    labels.append(item[2])

# 각 리스트를 넘파이 배열로 변환한다.
features = np.array(features)
labels = np.array(labels)

# 첫 번째 feature(입의 크기)의 분포를 시각화.
visualize_first_feature()

# 데이터 셋을 Normalize한다.
scaler = StandardScaler()
features = scaler.fit_transform(features)

# test set과 train set을 분리
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# knn 학습
knn = KNeighborsClassifier(n_neighbors=3)  # You can choose the value of k
knn.fit(X_train, y_train)
print("학습 중...")

# Predict and evaluate
y_pred = knn.predict(X_test)
