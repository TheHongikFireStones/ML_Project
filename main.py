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

# 눈 크기 측정
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
  
def visualize_feature(features, labels, feature_idx, feature_name):
    feature_values = [item[feature_idx] for item in features]
    unique_labels = list(set(labels))
    grouped_values = {label: [] for label in unique_labels}

    for value, label in zip(feature_values, labels):
        grouped_values[label].append(value)

    x_labels = []
    y_values = []

    for label in unique_labels:
        x_labels.append(label)
        y_values.append(grouped_values[label])

    box = plt.boxplot(y_values, labels=x_labels)
    plt.xlabel('Label')
    plt.ylabel(f'{feature_name} Value')
    plt.title(f'{feature_name}')

    # Calculate median values
    medians = [np.median(grouped_values[label]) for label in unique_labels]
    median_min = min(medians)
    median_max = max(medians)

    # Set y-axis limits around median values
    margin = (median_max - median_min) * 0.5  # Adjust margin for better visualization
    plt.ylim(median_min - margin, median_max + margin)

    # Set y-ticks around median values
    plt.yticks(np.arange(min(feature_values), max(feature_values) + 1, step=(max(feature_values) - min(feature_values)) / 10))

    # Add median values as annotations
    for i, line in enumerate(box['medians']):
        x, y = line.get_xydata()[1]  # top of median line
        plt.text(x, y, f'{y:.4f}', horizontalalignment='center', fontsize=8, color='black')
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
        try:
            visualize_feature(features, labels, idx, feature_name)
        except IndexError as e:
            print(f"Error: {e}. Index: {idx}, Feature Name: {feature_name}")

# google colab용 코드
from google.colab import drive
drive.mount('/content/drive')

def process_images(image_base_dir, label_dirs):
    data_set = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/content/drive/MyDrive/ML_Project_/shape_predictor_68_face_landmarks.dat")

    for label in label_dirs:
        label_path = os.path.join(image_base_dir, label)
        if not os.path.exists(label_path):
            print(f"Error: 디렉토리가 존재하지 않습니다. 경로를 확인하세요: {label_path}")
            continue

        for actor in os.listdir(label_path):
            actor_path = os.path.join(label_path, actor)
            if not os.path.isdir(actor_path):
                continue

            for filename in os.listdir(actor_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                file_path = os.path.join(actor_path, filename)
                if not os.path.isfile(file_path):
                    print(f"Error: 파일이 존재하지 않습니다. 경로를 확인하세요: {file_path}")
                    continue

                image = cv2.imread(file_path)
                if image is None:
                    print(f"Error: 이미지를 불러올 수 없습니다. 파일 형식을 확인하세요: {file_path}")
                    continue

                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    print(f"Error: cvtColor 함수에서 에러 발생: {e}")
                    continue

                faces = detector(gray)

                for face in faces:
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    landmarks = predictor(gray, face)

                    data_set.append([
                        get_mouth_width(x1, x2, landmarks),
                        get_face_slope_pair(landmarks),
                        get_eye_width(x1, x2, landmarks),
                        get_eyebrow_distance(y1, y2, landmarks),
                        get_nose_ratio(landmarks),
                        label  # Label is the folder name
                    ])

    return data_set

image_base_dir = '/content/drive/MyDrive/ML_Project_'
label_dirs = ["south_america", "bollywood_actor", "east_asian_actor", "white_man"]
data_set = process_images(image_base_dir, label_dirs)

# feature와 label을 각각 분리해 담을 리스트
features = []
labels = []

# feature와 label을 분리한다.
for item in data_set:
    feature_vector = item[:1] + list(item[1]) + item[2:5] #수염 feature 추가하면 같이 수정해야 함
    features.append(feature_vector)
    labels.append(item[5])  #feature 추가하면 6으로 바꿔야함

# 각 리스트를 넘파이 배열로 변환한다.
features = np.array(features)
labels = np.array(labels)

# feature들의 분포를 시각화.
visualize_all_features(features)

# 데이터 셋을 Normalize한다.
scaler = StandardScaler()
features = scaler.fit_transform(features)

# test set과 train set을 분리
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # knn 학습
# knn = KNeighborsClassifier(n_neighbors=3)  # You can choose the value of k
# knn.fit(X_train, y_train)
# print("학습 완료")

# 최적의 k값 찾기
for k in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print('k: %d, accuracy: %.2f' % (k, score*100))

# Predict and evaluate
y_pred = knn.predict(X_test)