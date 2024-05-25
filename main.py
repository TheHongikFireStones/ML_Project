import dlib
import cv2
import matplotlib.pyplot as plt
import boto3

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

# mustache, beard 특징을 감지하는 함수
def detect_faces(image):
    rekognition = boto3.client("rekognition")
    response = rekognition.detect_faces(
        Image={
            "S3Object": {
                "Bucket": "gprekognition",
                "Name": image,
            }
        },
        Attributes=['ALL'],
    )

    return response['FaceDetails']

def get_face_features(face_details):
    features = {}
    for detail in face_details:
        features['Mustache'] = 1 if detail['Mustache']['Value'] else 0
        features['Beard'] = 1 if detail['Beard']['Value'] else 0
    return features

# S3 버킷에서 이미지 리스트를 가져옵니다.
s3 = boto3.resource('s3')
bucket = s3.Bucket('gprekognition')

images = [obj.key for obj in bucket.objects.all()]

# 각 이미지에 대해 얼굴을 감지하고 특징을 출력합니다.
for image in images:
    face_details = detect_faces(image)
    features = get_face_features(face_details)
    print(f"{image} - Mustache: {features['Mustache']}, Beard: {features['Beard']}")