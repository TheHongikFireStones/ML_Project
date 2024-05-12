import dlib
import cv2
import matplotlib.pyplot as plt

MOUTH_LEFT_IDX = 48
MOUTH_RIGHT_IDX = 54


def get_mouth_width(face_left, face_right, landmarks):
    return (landmarks.part(MOUTH_RIGHT_IDX).x - landmarks.part(MOUTH_LEFT_IDX).x) / (face_right - face_left)


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
