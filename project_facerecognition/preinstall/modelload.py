import requests
import bz2
import os

# 파일 다운로드 함수
def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)

# Dlib 모델 파일 다운로드
landmarks_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
recognition_url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

landmarks_file = "shape_predictor_68_face_landmarks.dat.bz2"
recognition_file = "dlib_face_recognition_resnet_model_v1.dat.bz2"

# 다운로드 수행
download_file(landmarks_url, landmarks_file)
download_file(recognition_url, recognition_file)

# bzip2 압축 해제
def decompress_bz2(input_file, output_file):
    with bz2.BZ2File(input_file, 'rb') as f:
        with open(output_file, 'wb') as out:
            out.write(f.read())

# 압축 해제
decompress_bz2(landmarks_file, "shape_predictor_68_face_landmarks.dat")
decompress_bz2(recognition_file, "dlib_face_recognition_resnet_model_v1.dat")

# 다운로드된 파일 확인
print("다운로드 및 압축 해제가 완료되었습니다.")

