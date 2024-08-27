# -*- coding: utf-8 -*-
import boto3
import json
import uuid
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from IPython.display import display, Image
import mediapipe as mp
from ultralytics import YOLO
import time


# AWS 계정 및 리전 설정
aws_access_key_id = 'AKIA4MTWI2EHBACURLW6'
aws_secret_access_key = '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'
aws_region = 'ap-northeast-2'

# SQS 대기열 URL
sqs_queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/851725308174/v2q.fifo'
# Boto3 SQS 클라이언트 생성
sqs = boto3.client('sqs', region_name=aws_region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
unique_id = str(uuid.uuid4())

# 큐에 메시지 보내는 함수
def send_message_to_sqs(added_value):
    try:
        
        added_value['age'] = str(added_value['age'])
        response = sqs.send_message(
        QueueUrl=sqs_queue_url,
        MessageBody=json.dumps(added_value),
        MessageGroupId=unique_id,
        MessageDeduplicationId=unique_id
        )
        print(f"Message sent to SQS: {response['MessageId']}")
    except Exception as e:
        print(f"Error sending message to SQS: {e}")






# 모델 로드
face_model = YOLO("yolov8m-face.pt")
age_model = load_model('model1048_05_0.481.hdf5')
gender_model = load_model('model2_02_0.854.hdf5')

# 이미지 전처리
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# 나이 예측 모델
def predict_age(model, input_image):
    input_image = preprocess_image(input_image)
    age_prediction = model.predict(input_image)
    return age_prediction

# 성별 예측 모델
def predict_gender(model, input_image):
    input_image = preprocess_image(input_image)
    gender_prediction = model.predict(input_image)
    return gender_prediction
# 나이 라벨 리스트
age_labels = ['50', '40', '30', '20', '10']
predicted_gender_label = ['여','남']
def predict_age_gender(face_image):
    # 이미지를 모델에 입력할 형태로 변환하고 크기 조정
    face_image = cv2.resize(face_image, (224, 224))  # 모델이 기대하는 크기로 이미지 크기 조정
    face_image = np.expand_dims(face_image, axis=0)  # 배치 차원 추가
    face_image = face_image / 255.0  # 이미지 정규화

    # 나이 예측
    predicted_age = age_model.predict(face_image)
    max_age_index = predicted_age.argmax()  # 가장 높은 확률을 가진 나이 범주 인덱스 선택
    predicted_age_label = age_labels[max_age_index]  # 인덱스를 이용하여 라벨 선택

     # 성별 예측
    gender_prediction = gender_model.predict(face_image)
    max_gender_index = gender_prediction.argmax()
    predicted_gender = predicted_gender_label[max_gender_index]

    return predicted_age_label, predicted_gender
# 미디어 파이프
def detect_face_and_predict_age_gender(frame):
    face_count = 0
    predicted_info = []

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    multixVar = 0.1
    multiyUpVar = 0.34
    multiyDownVar = 0  

    

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                            max_num_faces=5,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.2) as face_mesh:

        # 얼굴 감지 및 예측 수행
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴 영역 추출
                xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
                # for landmark in face_landmarks.landmark:
                #     x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                #     xmin = min(xmin, x -30)
                #     ymin = min(ymin, y -120)
                #     xmax = max(xmax, x +30)
                #     ymax = max(ymax, y +30)
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    # 얼굴 영역의 좌표를 조정하여 화면 영역을 벗어나지 않도록 함
                    xmin = min(xmin, max(x , 0))  # x좌표에서 최소 30 이상
                    ymin = min(ymin, max(y , 0))  # y좌표에서 최소 120 이상
                    xmax = max(xmax, min(x , frame.shape[1]))  # x좌표에서 최대 frame 너비 이하
                    ymax = max(ymax, min(y , frame.shape[0]))  # y좌표에서 최대 frame 높이 이하
                face_image = frame[
                    max(int(ymin - ((ymax - ymin) * multiyUpVar)), 0):min(int(ymax + ((ymax - ymin) * multiyDownVar)), frame.shape[0]),
                    max(int(xmin - ((xmax - xmin) * multixVar)),  0):min(int(xmax + ((xmax - xmin) * multixVar)), frame.shape[1])
                ]
                # face_image= frame[ymin:ymax, xmin:xmax]

                # 나이와 성별 예측
                predicted_age, predicted_gender = predict_age_gender(face_image)

                # 얼굴 메시에 특징점 그리기
                mp_drawing.draw_landmarks(image=frame,
                                        landmark_list=face_landmarks,
                                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(image=frame,
                                        landmark_list=face_landmarks,
                                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(image=frame,
                                        landmark_list=face_landmarks,
                                        connections=mp_face_mesh.FACEMESH_IRISES,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                # 얼굴 메시 연결 라인 그리기
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in [10, 21, 54, 108, 132, 167, 355, 389, 405, 448, 491, 574]:  # 필요한 라인만 그리기
                        cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 1, (255, 0, 0), -1)

                # 예측 정보를 화면에 표시
                cv2.rectangle(frame, (max(int(xmin - ((xmax - xmin) * multixVar)),  0), max(int(ymin - ((ymax - ymin) * multiyUpVar)), 0)), (min(int(xmax + ((xmax - xmin) * multixVar)), frame.shape[1]), min(int(ymax + ((ymax - ymin) * multiyDownVar)), frame.shape[0])), (0, 255, 0), 2)
                # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, "Age: {}".format(predicted_age), (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "Gender: {}".format(predicted_gender), (xmin, ymin - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                predicted_info.append({'age': predicted_age, 'gender': predicted_gender})

    return face_count, predicted_info
file_path='jv11865229.jpg'


# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 일반적으로 기본 웹캠을 나타냄
count = 0

# 우선 순위 지정
priority = {
    ('여', 20): 1,
    ('여', 30): 2,
    ('남', 20): 3,
    ('여', 40): 4,
    ('남', 10): 5,
    ('여', 10): 6,
    ('남', 30): 7,
    ('여', 50): 8,
    ('남', 40): 9,
    ('남', 50): 10
}

# 성별, 나이, 눈맞춤 함수 실행 주기 설정
age_gender_execution_interval = 1  # 10초에 한 번씩 실행
last_age_gender_execution_time = time.time()
detected_objects = []
queue_arr = []

# 처음 객체가 인식되었는지 여부를 나타내는 변수
first_detection = True
# 이전 인식 시간을 저장하는 변수
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        # 프레임을 정상적으로 읽지 못한 경우, 계속 다음 프레임으로 진행
        continue
    
    # 얼굴을 감지하고 성별 및 나이를 예측하여 각 객체를 detected_objects에 추가
    face_count, predicted_info = detect_face_and_predict_age_gender(frame)

    if predicted_info:
        # 객체가 처음으로 인식되었거나 마지막 인식 이후 10초가 지났을 때에만 메시지를 전송
        if first_detection or time.time() - last_detection_time >= 29:
            # 각 얼굴에 대해 예측된 성별과 나이 정보를 우선 순위에 따라 큐에 추가
            for info in predicted_info:
                if info not in queue_arr:
                    queue_arr.append(info)
            queue_arr.sort(key=lambda x: priority.get((x['gender'], int(x['age'])), float('inf')))
            # SQS에 전송할 데이터 선택 (최우선 순위)
            if queue_arr:
                send_data = queue_arr[0]
                print("최종 전송 데이터:", send_data)
                send_message_to_sqs(send_data)
                # 메시지 전송 후에는 first_detection 변수를 False로 설정하여 다음에는 10초마다 수행되도록 함
                first_detection = False
                last_detection_time = time.time()
        else:
            first_detection
    else:
        print("인식된 객체가 없습니다.")
        # 객체가 인식되지 않았을 때는 first_detection을 다시 True로 설정
        first_detection = True
    
    # 웹캠 화면 표시
    cv2.imshow('WebCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 닫기
cap.release()
cv2.destroyAllWindows()
