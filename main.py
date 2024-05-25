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

LEFT_EYE_LEFT_IDX = 36
LEFT_EYE_RIGHT_IDX = 39

RIGHT_EYE_LEFT_IDX = 42
RIGHT_EYE_RIGHT_IDX = 45

LEFT_EYEBROW = 19
LEFT_EYE_IDX1 = 37
LEFT_EYE_IDX2 = 38

RIGHT_EYEBROW = 24
RIGHT_EYE_IDX1 = 43
RIGHT_EYE_IDX2 = 44

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

# 눈 크기 측정 함수
def get_left_eye_width(face_left, face_right, landmarks):
    return (landmarks.part(LEFT_EYE_RIGHT_IDX).x - landmarks.part(LEFT_EYE_LEFT_IDX).x) / (face_right - face_left)

def get_right_eye_width(face_left, face_right, landmarks):
    return (landmarks.part(RIGHT_EYE_RIGHT_IDX).x - landmarks.part(RIGHT_EYE_LEFT_IDX).x) / (face_right - face_left)

def get_eye_width(face_left, face_right, landmarks):
    left_eye_width = get_left_eye_width(face_left, face_right, landmarks)
    right_eye_width = get_right_eye_width(face_left, face_right, landmarks)
    average_width = (left_eye_width + right_eye_width) / 2
    return average_width

# 눈과 눈썹 사이 거리 측정 함수
def get_left_eyebrow_distance(face_top, face_bottom, landmarks):
    distance1 = (landmarks.part(LEFT_EYEBROW).y - landmarks.part(LEFT_EYE_IDX1).y) / (face_top - face_bottom)
    distance2 = (landmarks.part(LEFT_EYEBROW).y - landmarks.part(LEFT_EYE_IDX2).y) / (face_top - face_bottom)
    return min(distance1, distance2)

def get_right_eyebrow_distance(face_top, face_bottom, landmarks):
    distance1 = (landmarks.part(RIGHT_EYEBROW).y - landmarks.part(RIGHT_EYE_IDX1).y) / (face_top - face_bottom)
    distance2 = (landmarks.part(RIGHT_EYEBROW).y - landmarks.part(RIGHT_EYE_IDX2).y) / (face_top - face_bottom)
    return min(distance1, distance2)

def get_eyebrow_distance(face_top, face_bottom, landmarks):
    left_eyebrow = get_left_eyebrow_distance(face_top, face_bottom, landmarks)
    right_eyebrow = get_right_eyebrow_distance(face_top, face_bottom, landmarks)
    average_eyebrow_width = (left_eyebrow + right_eyebrow) / 2
    return average_eyebrow_width

# 코 관련 함수
def bezier_curve(p0, p1, p2, n_points=4):
    points = []
    for t in [i / (n_points - 1) for i in range(n_points)]:
        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        points.append((x, y))
    return points

def shoelace_formula(vertices):
    n = len(vertices)
    area = 0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2
        area -= y1 * x2

    area = abs(area) / 2.0
    return area

def get_nose_ratio(landmarks):
    # 코 교량의 양쪽 끝점
    nose_line1 = landmarks.part(27)
    nose_line2 = landmarks.part(28)
    nose_line3 = landmarks.part(29)
    nose_line4 = landmarks.part(30)

    nose_bottom1 = landmarks.part(31)
    nose_bottom2 = landmarks.part(32)
    nose_bottom3 = landmarks.part(33)
    nose_bottom4 = landmarks.part(34)
    nose_bottom5 = landmarks.part(35)

    p1 = (nose_bottom2.x, nose_line2.y)
    p2 = (nose_bottom4.x, nose_line2.y)

    p3 = (nose_bottom2.x, nose_line3.y)
    p4 = (nose_bottom4.x, nose_line3.y)

    p5 = (nose_bottom1.x, nose_line4.y)
    p6 = (nose_bottom5.x, nose_line4.y)

    nose_points = [
        (nose_line1.x, nose_line1.y),
        (nose_bottom1.x, nose_bottom1.y),
        (nose_bottom3.x, nose_bottom3.y),
        (nose_bottom5.x, nose_bottom5.y),
    ]

    nose_points.extend([p1, p2, p3, p4, p5, p6])

    nose_points.extend(bezier_curve(
        (nose_bottom1.x, nose_bottom1.y),
        (nose_line4.x, nose_line4.y),
        (nose_bottom3.x, nose_bottom3.y)
    )
    )

    nose_points.extend(bezier_curve(
        (nose_bottom5.x, nose_bottom5.y),
        (nose_line4.x, nose_line4.y),
        (nose_bottom3.x, nose_bottom3.y)
    )
    )

    face_points = []

    for i in range(17):
        face_points.append((landmarks.part(i).x, landmarks.part(i).y))

    for i in range(17, 22):
        face_points.append((landmarks.part(i).x, landmarks.part(i).y))

    for i in range(22, 27):
        face_points.append((landmarks.part(i).x, landmarks.part(i).y))

    nose_area = shoelace_formula(nose_points)
    face_area = shoelace_formula(face_points)

    return nose_area / face_area

def visualize_feature(features, feature_idx, feature_name):
    feature_values = [item[feature_idx] for item in features]
    x_values = np.random.normal(1, 0.01, len(feature_values))
    plt.scatter(x_values, feature_values, s=10)
    plt.xticks([1], [feature_name])
    plt.xlabel(feature_name)
    plt.ylabel('Value')
    plt.show()

def visualize_all_features(features):
    feature_names = [
        "Mouth Width",
        "Top Face Slope",
        "Bottom Face Slope",
        "Eye Width",
        "Eyebrow Distance",
        "Nose Ratio"
    ]

    for idx, feature_name in enumerate(feature_names):
        visualize_feature(features, idx, feature_name)

# 이미지 경로
image_dir = './img/'
# 디렉토리 이름 리스트
actors_list = ["Aditya_Roy_Kapur", "Arjun_Rampal", "Hrithik_Roshan", "John_Abraham", "Kartik_Aaryan", "Ranveer_Singh", "Shahid_Kapoor", "Sidharth_Malhotra", "Sidharth_Malhotra", "Varun_Dhawan"]

# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
                #눈 크기
                get_eye_width(x1, x2, landmarks),
                #눈과 눈썹 사이 거리
                get_eyebrow_distance(y1, y2, landmarks),
                # 코와 얼굴 비율
                get_nose_ratio(landmarks),

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
visualize_all_features(features)

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