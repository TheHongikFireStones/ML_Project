import dlib
import cv2
import matplotlib.pyplot as plt

# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이미지 로드
image = cv2.imread("backam3.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray_image)

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


for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 랜드마크 검출
    landmarks = predictor(gray_image, face)

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

    for (x, y) in nose_points:
        cv2.circle(image, (int(x), int(y)), 4, (255, 0, 0), -1)

    face_points = []

    for i in range(17):
       face_points.append((landmarks.part(i).x, landmarks.part(i).y))

    for i in range(17, 22):
       face_points.append((landmarks.part(i).x, landmarks.part(i).y))

    for i in range(22, 27):
       face_points.append((landmarks.part(i).x, landmarks.part(i).y))

    for (x, y) in face_points:
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

    result = shoelace_formula(nose_points)/shoelace_formula(face_points)

    print(round(result*100, 2))

# BGR 이미지를 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# matplotlib를 사용해 이미지 표시
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis("off")  # 축을 표시하지 않음
plt.show()



