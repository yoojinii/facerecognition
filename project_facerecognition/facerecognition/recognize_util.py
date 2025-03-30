import os
import cv2
import joblib
import tensorflow as tf
import dlib
from collections import defaultdict
from skimage import exposure
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import main as mc

# 사전 훈련된 모델 경로 설정
predictor_model = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\shape_predictor_68_face_landmarks.dat'
face_descriptor_model = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\dlib_face_recognition_resnet_model_v1.dat'
model_path = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\keras_face_recognition_model.keras'
scaler_path = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\scaler.pkl'
pca_svm_model_path = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\pca_svm_face_recognition_model.pkl'

# 실제 라벨 매핑
actual_labels = {
    0: '김우경',
    1: '김인서',
    2: '국소희',
    3: '안유진',
    4: '김채린'
}

final_folder = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\final_folder'  # 최종 이미지 저장 폴더

# 모델 및 스케일러 로드
pca_svm_model = joblib.load(pca_svm_model_path)
loaded_model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# 얼굴 감지 및 임베딩 추출 도구 설정
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_descriptor_extractor = dlib.face_recognition_model_v1(face_descriptor_model)

# 얼굴 인식 및 출석 수행 함수
def preprocess_face(image, face_rect, target_size=(160, 160)):
    x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
    cropped_face = image[y1:y2, x1:x2]
    equalized_face = exposure.equalize_adapthist(cropped_face, clip_limit=0.03)
    resized_face = cv2.resize(equalized_face, target_size)
    normalized_face = resized_face / 255.0
    return normalized_face

def extract_face_embedding(image, face_rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = face_pose_predictor(gray, face_rect)
    face_descriptor = face_descriptor_extractor.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# 가중 평균 예측 함수
def weighted_average_prediction(predictions_probs, threshold=0.85):
    weighted_sums = defaultdict(float)
    counts = defaultdict(int)

    for label, prob in predictions_probs:
        weighted_sums[label] += prob
        counts[label] += 1

    average_probs = {label: weighted_sums[label] / counts[label] for label in weighted_sums}
    best_label, best_avg_prob = max(average_probs.items(), key=lambda x: x[1])

    if best_avg_prob >= threshold:
        return actual_labels.get(best_label, "Unknown")
    else:
        return "결과 불충분"
    
def recognize_faces_in_images(image_folder, keras_model, svm_model, scaler, threshold=0.85, keras_threshold=0.9995):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    face_predictions = defaultdict(list)  # 얼굴별로 예측 결과를 저장
    recognized_students = []  # 인식된 학생의 학번을 저장

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"이미지 {image_file}을(를) 로드할 수 없습니다.")
            continue

        detected_faces = face_detector(image, 1)

        for i, face_rect in enumerate(detected_faces):
            face_embedding = extract_face_embedding(image, face_rect)

            # SVM 모델 예측
            prob_svm = svm_model.predict_proba([face_embedding])
            predicted_label_svm = svm_model.predict([face_embedding])[0]
            predicted_prob_svm = np.max(prob_svm)
            if predicted_prob_svm >= 0.85:
                face_predictions[i + 1].append((predicted_label_svm, predicted_prob_svm))
                print(f"Image: {image_file} | Face {i + 1} | SVM Predicted Name: {actual_labels[predicted_label_svm]} | Probability: {predicted_prob_svm:.4f}")

            # Keras 모델 예측
            face_embedding_scaled = scaler.transform([face_embedding])
            prob_keras = keras_model.predict(face_embedding_scaled)
            predicted_label_keras = np.argmax(prob_keras, axis=1)[0]
            predicted_prob_keras = np.max(prob_keras)
            if predicted_prob_keras >= 0.9995:
                face_predictions[i + 1].append((predicted_label_keras, predicted_prob_keras))
                print(f"Image: {image_file} | Face {i + 1} | Keras Predicted Name: {actual_labels[predicted_label_keras]} | Probability: {predicted_prob_keras:.4f}")


    for face_id, predictions_probs in face_predictions.items():
        final_result = weighted_average_prediction(predictions_probs)
        print(f"Face {face_id} 최종 결과: {final_result}")

        student_id = mc.get_student_id_by_name(final_result)

        # 최종 이미지에 BBox와 이름 표시
        image_with_results = draw_bounding_boxes_with_names(image, detected_faces, face_predictions)

        # 마지막 이미지라면 결과 저장
        if image_file == image_files[-1]:
            save_final_image_with_results(final_folder, image_file, image_with_results)
            
        if student_id is not None:
            recognized_students.append(student_id)
        else:
            print(f"{final_result}에 해당하는 학생을 찾을 수 없습니다.")

    return recognized_students  # 인식된 학생들의 학번 리스트 반환



def draw_bounding_boxes_with_names(image, face_rects, face_predictions):
    image_with_results = image.copy()

    # 한글을 출력할 폰트 로드
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)  # Windows에서 Malgun 폰트 사용
    except IOError:
        font = ImageFont.load_default()  # 폰트 로드 실패 시 기본 폰트 사용

    pil_image = Image.fromarray(cv2.cvtColor(image_with_results, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for i, face_rect in enumerate(face_rects):
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        cv2.rectangle(image_with_results, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 초록색 BBox

        if face_predictions.get(i + 1):
            final_result = weighted_average_prediction(face_predictions[i + 1])
            text = f"{final_result}"
            draw.text((x, y - 10), text, font=font, fill=(0, 255, 0))  # 이름 출력

    image_with_results = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image_with_results

def save_final_image_with_results(final_folder, image_file, image_with_results):
    output_path = os.path.join(final_folder, f"final_result_{image_file}")
    cv2.imwrite(output_path, image_with_results)
    print(f"최종 결과 이미지 저장: {output_path}")
