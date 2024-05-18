import dlib
import cv2
import matplotlib.pyplot as plt

MOUTH_LEFT_IDX = 48
MOUTH_RIGHT_IDX = 54


def get_mouth_width(face_left, face_right, landmarks):
    return (landmarks.part(MOUTH_RIGHT_IDX).x - landmarks.part(MOUTH_LEFT_IDX).x) / (face_right - face_left)


def get_face_slope_pair(landmarks):
    left_top_slope = abs(landmarks.part(0).y - landmarks.part(4).y) / abs(landmarks.part(0).x - landmarks.part(4).x)
    right_top_slope = abs(landmarks.part(16).y - landmarks.part(12).y) / abs(landmarks.part(16).x - landmarks.part(12).x)

    left_bottom_slope = abs(landmarks.part(5).y - landmarks.part(8).y) / abs(landmarks.part(5).x - landmarks.part(8).x)
    right_bottom_slope = abs(landmarks.part(11).y - landmarks.part(8).y) / abs(landmarks.part(11).x - landmarks.part(8).x)

    top_slope_avg = (left_top_slope + right_top_slope) / 2
    bottom_slope_avg = (left_bottom_slope + right_bottom_slope) / 2
    return top_slope_avg, bottom_slope_avg


# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
    print(get_mouth_width(x1, x2, landmarks))
    print(get_face_slope_pair(landmarks))
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        if n in (0, 4, 12, 16, 5, 8, 11):
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

# BGR 이미지를 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# matplotlib를 사용해 이미지 표시
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis("off")  # 축을 표시하지 않음
plt.show()
