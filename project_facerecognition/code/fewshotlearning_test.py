import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import os

# ArcFace와 유사한 사전학습된 얼굴 인식 모델 사용
arcface_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False)

# 사용자 데이터 파일 경로
USER_DATA_FILE = 'user_data.pth'

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 사용자 정보 로드 (파일이 있는 경우)
if os.path.exists(USER_DATA_FILE):
    user_data = torch.load(USER_DATA_FILE)
    user_embeddings = user_data['embeddings']
    user_names = user_data['names']
else:
    user_embeddings = []
    user_names = []

# 얼굴 임베딩 계산 함수
def get_face_embedding(model, face_image):
    with torch.no_grad():
        embedding = model(face_image.unsqueeze(0))
    return F.normalize(embedding, p=2, dim=1)

# 사용자 등록
def register_user():
    cap = cv2.VideoCapture(1)
    
    print("등록할 사용자의 이름을 입력하세요:")
    user_name = input("이름: ")
    
    if user_name in user_names:
        print("이미 등록된 이름입니다. 다른 이름을 입력하세요.")
        cap.release()
        return

    embeddings = []

    while len(embeddings) < 4:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 영상을 불러올 수 없습니다.")
            break

        # 얼굴 검출
        boxes, probs = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                # 얼굴 영역을 잘라서 처리
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                pil_image = Image.fromarray(face)
                query_image = transform(pil_image)

                # 임베딩 계산
                face_embedding = get_face_embedding(arcface_model, query_image)
                embeddings.append(face_embedding)
                print(f"{len(embeddings)}/4 사진 등록 완료")

                # 얼굴 표시
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"Registered: {user_name}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 사용자 데이터 저장
    if len(embeddings) == 4:
        user_embeddings.extend(embeddings)
        user_names.extend([user_name] * 4)
        torch.save({'embeddings': user_embeddings, 'names': user_names}, USER_DATA_FILE)
        print(f"{user_name}님이 등록되었습니다.")
    else:
        print("등록에 실패했습니다. 4장 모두 찍어야 합니다.")

    cap.release()
    cv2.destroyAllWindows()

# 등록된 사용자와 일치 여부 확인
def predict_user(frame):
    boxes, probs = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            pil_image = Image.fromarray(face)
            query_image = transform(pil_image)

            # 얼굴 임베딩 계산
            query_embedding = get_face_embedding(arcface_model, query_image)

            # 등록된 사용자와 거리 계산
            match_counts = {}
            for name, embedding in zip(user_names, user_embeddings):
                distance = torch.dist(query_embedding, embedding).item()
                if distance < 0.6:
                    match_counts[name] = match_counts.get(name, 0) + 1
            
            matched_name = "Unknown"
            match_score = 0

            # 3회 이상 일치한 경우 가장 많이 일치한 이름으로 설정
            if match_counts:
                matched_name = max(match_counts, key=match_counts.get)
                match_score = match_counts[matched_name] / 4.0  # 일치도 계산
                if match_counts[matched_name] < 3:
                    matched_name = "Unknown"  # 최소 3개 일치하지 않으면 Unknown

            # 결과 화면에 표시
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"Detected: {matched_name} ({match_score:.2f})", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Webcam Recognition", frame)

# 메인 함수
cap = cv2.VideoCapture(1)

print("사용자 등록: 'r' 키를 누르세요. 종료하려면 'q'를 누르세요.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 키 입력 확인
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        cap.release()  # 웹캠 종료 후 등록 모드로 전환
        register_user()
        cap = cv2.VideoCapture(1)  # 웹캠 다시 시작
    elif key == ord('q'):
        break
    else:
        predict_user(frame)  # 인식 모드

cap.release()
cv2.destroyAllWindows()
