import dlib
import cv2
import matplotlib.pyplot as plt

# 눈 크기 측정

LEFT_EYE_LEFT_IDX = 36
LEFT_EYE_RIGHT_IDX = 39

RIGHT_EYE_LEFT_IDX = 42
RIGHT_EYE_RIGHT_IDX = 44

def get_left_eye_width(face_left, face_right, landmarks):
    return (landmarks.part(LEFT_EYE_RIGHT_IDX).x - landmarks.part(LEFT_EYE_LEFT_IDX).x) / (face_right - face_left)

def get_right_eye_width(face_left, face_right, landmarks):
    return (landmarks.part(RIGHT_EYE_RIGHT_IDX).x - landmarks.part(RIGHT_EYE_LEFT_IDX).x) / (face_right - face_left)

def get_eye_width(face_left, face_right, landmarks):
    left_eye_width = get_left_eye_width(face_left, face_right, landmarks)
    right_eye_width = get_right_eye_width(face_left, face_right, landmarks)
    average_width = (left_eye_width + right_eye_width) / 2
    return average_width

# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

from google.colab import drive
drive.mount('/content/drive')

# 이미지 로드
image = cv2.imread("asian_image/1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray)

for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 랜드마크 검출
    landmarks = predictor(gray, face)
    print(get_eye_width(x1, x2, landmarks))
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

# BGR 이미지를 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# matplotlib를 사용해 이미지 표시
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis("off")  # 축을 표시하지 않음
plt.show()