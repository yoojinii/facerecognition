from flask import Flask, render_template, redirect, url_for, request, Response, flash, session, jsonify
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date, datetime, timedelta 
from sklearn.preprocessing import StandardScaler
from PIL import ImageFont, ImageDraw, Image
import dlib
import sqlite3
import re
import traceback
import asyncio
import websockets
import numpy as np
import tensorflow as tf
import joblib
import cv2
import recognize_util as ru
import os
import time  
from dlib import correlation_tracker  


app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = '1234'

# 비디오 캡처 초기화
video_capture = cv2.VideoCapture(1)  # 기본 카메라 사용
video_capture = None
stream_active = False
capture_active = False

# 사전 훈련된 모델 경로 설정
predictor_model = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\shape_predictor_68_face_landmarks.dat'
face_descriptor_model = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\dlib_face_recognition_resnet_model_v1.dat'
model_path = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\keras_face_recognition_model.keras'
scaler_path = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\scaler.pkl'
pca_svm_model_path = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\pca_svm_face_recognition_model.pkl'

# dlib의 얼굴 감지기 및 포즈 예측기 초기화
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1(r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\classified_model\dlib_face_recognition_resnet_model_v1.dat')

# 모델 및 스케일러 로드
pca_svm_model = joblib.load(pca_svm_model_path)
loaded_model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# 실제 라벨 매핑
actual_labels = {
    0: '김우경',
    1: '김인서',
    2: '국소희',
    3: '안유진',
    4: '김채린'
}

#전역변수
video_capture = None
stream_active = False
capture_active = False
track_running=False
is_running=True 
start_time = None # 캠 시작 시간을 저장할 변수
tracker = correlation_tracker()
tracked_students = {}  # 학생 이름과 트래킹 상태를 저장하는 딕셔너리
TRACK_TIMEOUT = 10  # 얼굴 인식이 안 된 후 경과 시간 (10초로 설정)

# 트레킹 전역 변수 정의
tracking_students = {"present": [], "late": []}  # 출석, 지각 대상 학생
tracking_active = False  # 트레킹 모드 활성화 상태
missing_students = []  # 감지되지 않은 학생 목록
resolved_students = set()  # 버튼 처리가 완료된 학생을 저장

# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
try:
    font = ImageFont.truetype(font_path, 20)
except IOError:
    print(f"폰트를 찾을 수 없습니다: {font_path}")
    exit()

@app.route('/')
def home():
    # 로그인 상태를 확인하는 조건을 추가하세요.
    is_logged_in = False  # 이 값을 로그인 상태에 따라 True 또는 False로 설정하세요.

    if is_logged_in:
        return render_template('dashboard.html')  # 로그인한 경우 대시보드로 리다이렉트
    else:
        return render_template('login.html')  # 로그인하지 않은 경우 로그인 페이지 렌더링

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        studentID = request.form['studentID']

        # 중복 이메일 및 아이디 확인
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ? OR studentID = ?', (email, studentID))
        account = cursor.fetchone()

        if account:
            return '이미 등록된 이메일 또는 아이디입니다.'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            return '유효하지 않은 이메일 형식입니다.'
        elif not username or not password or not email or not studentID:
            return '모든 필드를 입력해주세요.'
        else:
            # 비밀번호 해시화 후 저장
            hashed_password = generate_password_hash(password)
            cursor.execute('INSERT INTO users (username, password, email, studentID) VALUES (?, ?, ?, ?)',
                           (username, hashed_password, email, studentID))
            sqlite3.connection.commit()
            cursor.close()
            return redirect(url_for('login'))

    return render_template('register.html')

# 로그인 라우트
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'studentID' in request.form and 'password' in request.form:
        studentID = request.form['studentID']
        password = request.form['password']

        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM users WHERE studentID = ?', (studentID,))
        account = cursor.fetchone()

        if account and account['password']==password:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['is_admin'] = account['is_admin'] #is_admin = 0 is student 1 is professor
            session['studentID'] = account['studentID']
            return redirect(url_for('dashboard'))
        else:
            print(f"Queried account: {dict(account)}")
            return '로그인 실패'
    return render_template('login.html')

# 대시보드 라우트
@app.route('/dashboard')
def dashboard():
    if 'loggedin' in session:
        # 사용자 수강 과목 정보를 불러오기 (중복 제거)
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        username= session['username']
        is_admin=session['is_admin']
        if is_admin==1:
            cursor.execute('''
                SELECT DISTINCT classes.class_id,classes.class_name, classes.section
                FROM classes
                INNER JOIN users ON classes.professor_id = users.id
                WHERE users.id = ?
        ''', (session['id'],))
        elif is_admin==0:
            cursor.execute('''
                SELECT DISTINCT classes.class_id,classes.class_name, classes.section
                FROM classes
                INNER JOIN enrollment ON classes.class_id = enrollment.class_id
                INNER JOIN users ON enrollment.student_id =users.studentID
                WHERE users.studentID = ?
            ''', (session['studentID'],))
        classes = cursor.fetchall()
        conn.close()
        return render_template('dashboard.html', classes=classes, username=username, is_admin=is_admin, active_page='dashboard')
    return redirect(url_for('login'))


# 계정 페이지 라우트
@app.route('/account')
def account():
    if 'loggedin' in session:
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # 로그인한 사용자의 정보를 가져옴
        cursor.execute('SELECT * FROM users WHERE id = ?', (session['id'],))
        user_info = cursor.fetchone()

        # 사용자가 수강 중인 과목 정보를 가져옴
        cursor.execute('''
            SELECT c.class_id, c.class_name, c.section 
            FROM enrollment e
            JOIN classes c ON e.class_id = c.class_id
            JOIN users u ON e.student_id = u.studentID
            WHERE u.studentID = ?
        ''', (session['studentID'],))
        enrolled_classes = cursor.fetchall()

        return render_template('account.html', user_info=user_info, enrolled_classes=enrolled_classes,
                               active_page='account')
    return redirect(url_for('login'))

# 출석 기록 삽입 or 갱신 라우트
def record_attendance(student_id, class_id, status="출석"):
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()

    # 동일한 student_id와 class_id에 대한 출석 기록이 있는지 확인
    cursor.execute('''
        SELECT * FROM attendance WHERE student_id = ? AND class_id = ? AND attendance_date = CURRENT_DATE
    ''', (student_id, class_id))
    
    existing_record = cursor.fetchone()
    
    if not existing_record:  # 기록이 없으면 출석 추가
        cursor.execute('''
            INSERT INTO attendance (student_id, class_id, status)
            VALUES (?, ?, ?)
        ''', (student_id, class_id, status))
        print(f"{student_id} 학생의 출석 상태를 {status}로 기록.")
        conn.commit()
    else:
        # 이미 출석이 기록되어 있을 경우, 출석 상태가 다르면 업데이트
        if existing_record[3] != status:  # 기존 상태가 "출석"이 아니면 상태 변경
            cursor.execute('''
                UPDATE attendance SET status = ? WHERE student_id = ? AND class_id = ? AND attendance_date = CURRENT_DATE
            ''', (status, student_id, class_id))
            print(f"{student_id} 학생의 출석 상태를 {status}로 기록.")
            conn.commit()
        else:
            print(f"{student_id} 학생의 출석 상태가 이미 {status}로 기록되어 있습니다.")
    
    conn.close()


def get_student_id_by_name(name):
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()
    cursor.execute("SELECT studentID FROM users WHERE username = ?", (name,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]  # studentID
    else:
        return None  # 이름이 없을 경우
    

def get_current_class_id():
    class_id = request.args.get('class_id', type=int)
    if class_id is None:
        print("Error: No class_id provided in request.")
    return class_id


# 실시간 비디오 스트리밍 함수
def generate_video_feed(capture_flag=False,num_images=15,output_dir='C:\\Users\\youji\\OneDrive\\project_facerecognition\\project_facerecognition\\facerecognition\\captured_images', class_id=None):
    print("generate_video_feed시작")
    global video_capture, stream_active, capture_active, track_running
    video_capture = cv2.VideoCapture(1) 
    capture_count = 0
    captured_images = []


    if capture_flag and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while stream_active:
        ret, frame = video_capture.read()
        if not ret:
            continue
        
        # 트래킹이 활성화된 경우 generate_frames로 처리
        if track_running:
            yield from generate_frames(class_id)
        else:
            # OpenCV로 이미지를 JPEG 형식으로 인코딩하여 스트리밍
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            if capture_active:
                image_path = os.path.join(output_dir, f"img_{capture_count+1}.jpg")
                cv2.imwrite(image_path, frame)
                captured_images.append(image_path)
                capture_count += 1

            if capture_active and capture_count >= num_images:
                capture_active = False


@socketio.on('connect')
def handle_connect():
    print("클라이언트가 연결되었습니다.")

@app.route('/check_student_status')
def check_student_status():
    # tracked_students에서 10초 동안 인식되지 않은 학생 확인
    alert_message = None
    current_time = time.time()

    for student, last_seen_time in list(tracked_students.items()):
        if current_time - last_seen_time > TRACK_TIMEOUT:
            alert_message = f"{student} 학생이 10초 동안 인식되지 않았습니다."
            del tracked_students[student]  # 추적 목록에서 제거
            break  # 첫 번째로 인식되지 않은 학생에 대해서만 경고

    if alert_message:
            socketio.emit('alert', {'message': alert_message})

  

def get_enrolled_students(class_id):
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()
    cursor.execute("SELECT student_id FROM enrollment WHERE class_id = ?", (class_id,))
    enrolled_students = cursor.fetchall()
    conn.close()
    return [student[0] for student in enrolled_students]


def generate_frames(class_id):
    global is_running, start_time, tracker, tracked_students, student_tracking_status
    
    if not video_capture.isOpened():
        print("카메라를 열 수 없습니다")
        return

    # 캠 시작 시간을 기록합니다.
    start_time = time.time()

    # 수업에 등록된 모든 학생을 불러옵니다.
    enrolled_students = get_enrolled_students(class_id)
    
    # 학생 트래킹 상태를 초기화합니다.
    student_tracking_status = {student: {'last_seen': 0, 'status': '미인식'} for student in enrolled_students}

    while is_running:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            current_time = time.time()

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = face_detector(rgb_image, 1)

            if len(detected_faces) > 0:
                for face_rect in detected_faces:  # 모든 얼굴에 대해 반복
                    x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

                    # 얼굴 특징점 예측 및 트래킹
                    shape = face_pose_predictor(rgb_image, face_rect)
                    face_descriptor = face_descriptor_extractor.compute_face_descriptor(rgb_image, shape)
                    embedding = np.array(face_descriptor).reshape(1, -1)

                    new_embedding_scaled = scaler.transform(embedding)
                    prediction_prob = loaded_model.predict(new_embedding_scaled)
                    predicted_class = np.argmax(prediction_prob, axis=1)[0]
                    confidence = prediction_prob[0][predicted_class]
                    label = actual_labels.get(predicted_class, "Unknown")

                    if label != "Unknown":
                        tracked_students[label] = current_time
                        student_id = get_student_id_by_name(label)

                        if student_id in student_tracking_status:
                            student_tracking_status[student_id]['last_seen'] = current_time
                            student_tracking_status[student_id]['status'] = '출석'
                            record_attendance(student_id, class_id, "출석")

                    # 얼굴 영역에 박스를 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label}: {confidence:.2f}"
                    pil_image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((x1 + 2, y1 - 28), text, font=font, fill=(0, 0, 0, 255))
                    frame = np.array(pil_image)

            # 모든 등록된 학생의 출석 상태 확인
            for student_id, status in student_tracking_status.items():
                last_seen = status['last_seen']
                time_elapsed = current_time - last_seen
                
                if time_elapsed > 40:  # 40초 이상 인식되지 않으면 결석 처리
                    if status['status'] != '결석':
                        student_tracking_status[student_id]['status'] = '결석'
                        record_attendance(student_id, class_id, "결석")
                        alert_message = f"{student_id} 학생이 40초 동안 인식되지 않았습니다. 결석처리"
                        socketio.emit('alert', {'message': alert_message})
                elif time_elapsed > 20:  # 20초 이상 인식되지 않으면 지각 처리
                    if status['status'] != '지각':
                        student_tracking_status[student_id]['status'] = '지각'
                        record_attendance(student_id, class_id, "지각")
                        alert_message = f"{student_id} 학생이 20초 동안 인식되지 않았습니다. 지각처리"
                        socketio.emit('alert', {'message': alert_message})
                elif time_elapsed <= 10:  # 10초 이내에 인식되었으면 출석 유지
                    student_tracking_status[student_id]['status'] = '출석'

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()



# 비디오 스트리밍 라우트
@app.route('/video_feed')
def video_feed():
    print("video_feed시작")
    global stream_active
    stream_active = True
    class_id = request.args.get('class_id')
    week = request.args.get('week')

    return Response(generate_video_feed(class_id=class_id), mimetype='multipart/x-mixed-replace; boundary=frame')


# 출석 관리 라우트
@app.route('/attendance/<int:class_id>/<int:week>', methods=['GET', 'POST'])
def attendance(class_id,week):
    if 'loggedin' in session:
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if request.method == 'POST':
            # 교수만 출석 상태 업데이트 가능
            if session['is_admin']:  # 교수인지 확인하는 부분, is_admin 플래그를 통해 구분
                student_id = request.form['student_id']
                status = request.form['status']
                current_date = date.today()  # 현재 날짜 가져오기

                cursor.execute('''
                    INSERT INTO attendance (student_id, class_id, status, attendance_date)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(student_id, class_id) DO UPDATE SET
                        status = excluded.status,
                        attendance_date = excluded.attendance_date
                ''', (student_id, class_id, status, current_date))
                sqlite3.connection.commit()

        # 수업의 전체 학생 출석 현황 조회
        cursor.execute('''
            SELECT u.username, a.status, a.student_id, a.attendance_date
            FROM users u
            LEFT JOIN attendance a ON u.studentID = a.student_id AND a.class_id = ?
            WHERE u.studentID IN (SELECT student_id FROM enrollment WHERE class_id = ?)
        ''', (class_id, class_id))
        students = cursor.fetchall()

        # 수업의 정보를 조회 (선택적으로 표시)
        cursor.execute('SELECT * FROM classes WHERE class_id = ?', (class_id,))
        class_info = cursor.fetchone()

        return render_template('attendance.html', students=students, class_info=class_info, class_id=class_id,week=week,
                               active_page='attendance')

    return redirect(url_for('login'))


@app.route('/get_attendance')
def get_attendance():
    class_id = request.args.get('class_id')
    date = datetime.today().strftime('%Y-%m-%d')
    # 출석한 학생의 이름을 리스트 형태로 반환
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT username 
                   FROM attendance JOIN users ON attendance.student_id = users.studentID 
                   WHERE status = "출석" 
                   AND attendance.class_id = ? 
                   AND attendance_date = ?''',(class_id,date))
    students = cursor.fetchall()
    conn.close()

    # 학생 이름을 그대로 반환
    student_names = list(set(student[0] for student in students))
    return jsonify(student_names)

# 얼굴 인식 및 출석 수행 함수
def perform_attendance(class_id, week):
    image_folder = 'captured_images'
    recognized_students=ru.recognize_faces_in_images(image_folder, loaded_model, pca_svm_model, scaler, class_id, week)
    current_date = date.today()  # 현재 날짜
    for student_id in recognized_students:
        record_attendance(student_id, class_id, status="출석")

# 출석 시작 라우트
@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    global capture_active
    data = request.get_json()
    class_id = data.get('class_id')
    week = data.get('week')
    
    capture_active = True
    perform_attendance(class_id, week)  # 캡처 및 얼굴 인식 수행
    return jsonify({"status": "attendance started"})

# 출석 종료 라우트 
@app.route('/end_attendance', methods=['POST'])
def end_attendance():
    global stream_active, video_capture
    # 출석 종료 처리 로직
    data = request.get_json()
    class_id = data.get('class_id')
    week = data.get('week')

    # 스트림 종료
    stream_active = False
    if video_capture:
        video_capture.release()  # 웹캠 해제
        video_capture = None

    return jsonify({"status": "Attendance ended", "class_id": class_id, "week": week})

# 출결 현황 조회 라우트
@app.route('/attendancecheck/<int:class_id>', methods=['GET'])
def attendancecheck(class_id):
    if 'loggedin' in session:
        # 로그인한 학생의 ID 가져오기
        student_id = session['studentID']
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # 해당 수업의 정보 및 교수 이름 가져오기
        cursor.execute('''
            SELECT c.class_name, u.username AS professor_name
            FROM classes c
            JOIN users u ON c.professor_id = u.id
            WHERE c.class_id = ?
        ''', (class_id,))
        class_info = cursor.fetchone()
        class_name = class_info['class_name'] if class_info else 'Unknown'
        professor_name = class_info['professor_name'] if class_info else 'Unknown'
        # 해당 과목의 학생 출석 현황 불러오기
        cursor.execute('''
                       select class_name
                       From classes
                       Where class_id =?
                       ''', (class_id,))
        class_info = cursor.fetchone()
        class_name = class_info['class_name'] if class_info else 'Unknown'
        cursor.execute('''
            SELECT strftime('%W',a.attendance_date) as week, a.class_id, a.student_id, u.username, a.status, a.attendance_date
            FROM attendance a
            JOIN users u ON a.student_id = u.studentID
            WHERE a.class_id = ? AND a.student_id = ?
            ORDER BY a.attendance_date ASC
        ''', (class_id,student_id))
        attendance_records = cursor.fetchall()
        semester_start_date=datetime(2024,10,13)
        attendance_by_week = {}
        for record in attendance_records:
            attendance_date = datetime.strptime(record['attendance_date'], '%Y-%m-%d')
            # 학기 시작일로부터 몇 번째 주인지 계산
            week = ((attendance_date - semester_start_date).days // 7) + 1
            if week not in attendance_by_week:
                attendance_by_week[week] = []
            attendance_by_week[week].append(record)

        attendance_count = sum(1 for record in attendance_records if record['status']=='출석')
        tardy_count = sum(1 for record in attendance_records if record['status']=='지각')
        absence_count = sum(1 for record in attendance_records if record['status']=='결석')
        context = {
            'class_name' : class_name,
            'professor_name' : professor_name,
            'attendance_count' : attendance_count,
            'tardy_count' : tardy_count,
            'absence_count' : absence_count,
            'unknown_count' : 0,
            'attendance_by_week' : attendance_by_week,
            'attendance_records': attendance_records
        }
        conn.close()
        return render_template('attendancecheck.html', **context)
    return redirect(url_for('login'))

# 캘린더 페이지 라우트
@app.route('/calendar')
def calendar():
    if 'loggedin' in session:
        return render_template('calendar.html', active_page='calendar')
    return redirect(url_for('login'))

# 일정 추가 API
@app.route('/add_event', methods=['POST'])
def add_event():
    if 'studentID' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    data = request.get_json()
    title = data.get('title')
    start = data.get('start')
    end = data.get('end')
    studentID = session['studentID']  # 현재 로그인한 사용자의 studentID

    # 입력값 유효성 검사
    if not title or not start:
        return jsonify({'status': 'error', 'message': 'Title and start date are required'}), 400

    print(f"Received data - Title: {title}, Start: {start}, End: {end}, StudentID: {studentID}")  # 데이터 확인용

    try:
        conn = sqlite3.connect('university.db')
        print("Database connection established")  # 연결 확인
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (title, start_date, end_date, studentID) VALUES (?, ?, ?, ?)
        ''', (title, start, end, studentID))
        conn.commit()
        new_event_id = cursor.lastrowid  # 추가된 이벤트의 ID를 가져옴
        conn.close()

        print("Event added successfully with ID:", new_event_id)  # 이벤트 추가 확인
        return jsonify({'status': 'success', 'id': new_event_id}), 200
    except Exception as e:
        print("Error adding event:", e)
        traceback.print_exc()  # 전체 오류 스택 출력
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 일정 불러오기 API
@app.route('/get_events')
def get_events():
    if 'studentID' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    studentID = session['studentID']  # 현재 로그인한 사용자 ID

    try:
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT rowid as id, title, start_date as start, end_date as end FROM events WHERE studentID = ?', (studentID,))
        events = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(events)
    except Exception as e:
        print("Error fetching events:", e)  # 오류 로그 출력
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 일정 삭제 API
@app.route('/delete_event/<int:event_id>', methods=['DELETE'])
def delete_event(event_id):
    if 'studentID' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    studentID = session['studentID']  # 현재 로그인한 사용자 ID

    try:
        conn = sqlite3.connect('university.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM events WHERE rowid = ? AND studentID = ?', (event_id, studentID))
        conn.commit()
        conn.close()

        print("Event deleted successfully with ID:", event_id)  # 삭제 확인
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print("Error deleting event:", e)  # 오류 로그 출력
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 일정 업데이트 API
@app.route('/update_event/<int:event_id>', methods=['PUT'])
def update_event(event_id):
    if 'studentID' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    data = request.get_json()
    title = data.get('title')
    start = data.get('start')
    end = data.get('end')
    studentID = session['studentID']  # 현재 로그인한 사용자 ID

    print(f"Updating event ID: {event_id} - Title: {title}, Start: {start}, End: {end}, StudentID: {studentID}")  # 데이터 확인용

    try:
        conn = sqlite3.connect('university.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE events
            SET title = ?, start_date = ?, end_date = ?
            WHERE rowid = ? AND studentID = ?
        ''', (title, start, end, event_id, studentID))
        conn.commit()
        conn.close()

        print("Event updated successfully with ID:", event_id)  # 업데이트 확인
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print("Error updating event:", e)  # 오류 로그 출력
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 일정 목록을 가져오는 API
@app.route('/get_upcoming_events')
def get_upcoming_events():
    if 'studentID' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    studentID = session['studentID']  # 현재 로그인한 사용자의 studentID
    current_date = datetime.now().strftime('%Y-%m-%d')
    one_week_from_now = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

    try:
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 현재 날짜 이후의 일정을 가져오되, 가까운 순서로 정렬
        cursor.execute('''
            SELECT rowid as id, title, start_date, end_date 
            FROM events 
            WHERE studentID = ? AND start_date >= ?
            ORDER BY start_date ASC
            LIMIT 10
        ''', (studentID, current_date))

        events = []
        for row in cursor.fetchall():
            event = dict(row)
            # 긴급 알림 확인 (일주일 이내인지 확인)
            event['is_urgent'] = event['start_date'] <= one_week_from_now
            events.append(event)

        conn.close()
        return jsonify({'status': 'success', 'events': events}), 200
    except Exception as e:
        print("Error fetching events:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
    

# 학생 목록 조회 라우트
@app.route('/studentlist/<int:class_id>/<int:week>')
def studentlist(class_id, week):
    print(f'Class ID: {class_id}, Week: {week}')
    if 'loggedin' in session:
        conn = sqlite3.connect('university.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(''' 
            SELECT u.studentID, u.username, a.status, a.attendance_date
            FROM users u
            INNER JOIN enrollment e ON u.studentID = e.student_id
            LEFT JOIN attendance a ON u.studentID = a.student_id AND e.class_id = a.class_id
            WHERE e.class_id = ?
        ''', (class_id,))
        students = cursor.fetchall()
        
        semester_start_date = datetime(2024, 10, 13)
        attendance_by_week = {}
        for student in students:
            if student['attendance_date']:  # attendance_date가 존재할 때
                attendance_date = datetime.strptime(student['attendance_date'], '%Y-%m-%d')
                attendance_week = ((attendance_date - semester_start_date).days // 7) + 1
                if attendance_week not in attendance_by_week:
                    attendance_by_week[attendance_week] = []
                attendance_by_week[attendance_week].append(student)
        filtered_students = attendance_by_week.get(week, [])
        
        cursor.execute('SELECT class_name FROM classes WHERE class_id = ?', (class_id,))
        class_info = cursor.fetchone()
        class_name = class_info['class_name'] if class_info else 'Unknown'
        conn.close()

        return render_template('studentlist.html', students=filtered_students, class_name=class_name, week=week, class_id=class_id, active_page='studentlist')
    
    return redirect(url_for('login'))


# 출결 수정 라우트 
@app.route('/update_attendance/<int:class_id>', methods=['POST'])
def update_attendance(class_id):
    if 'loggedin' in session:
        student_id = request.form['student_id']
        status = request.form['status']
        
        # week 값이 문자열로 넘어오므로 이를 정수로 변환
        week = int(request.form['week'])  # 여기에 정수로 변환

        # 출석 상태 수정
        conn = sqlite3.connect('university.db')
        cursor = conn.cursor()

        # 주차에 맞는 날짜를 계산하여 해당 주차의 출석 상태를 수정
        semester_start_date = datetime(2024, 10, 13)
        start_of_week = semester_start_date + timedelta(weeks=week-1)  # week-1은 정상 작동
        end_of_week = start_of_week + timedelta(days=6)
        
        # 해당 학생의 출석 상태 업데이트
        cursor.execute(''' 
            UPDATE attendance
            SET status = ?
            WHERE student_id = ? AND class_id = ? AND attendance_date BETWEEN ? AND ?
        ''', (status, student_id, class_id, start_of_week.date(), end_of_week.date()))

        conn.commit()
        conn.close()

        # 플래시 메시지
        flash(f"{student_id} 학생의 출석 상태가 {status}로 변경되었습니다.", "success")

        # 수정 후 다시 학생 목록 페이지로 리다이렉트
        return redirect(url_for('studentlist', class_id=class_id, week=week))

    return redirect(url_for('login'))


# 메시지 라우트
@app.route('/message', methods=['GET'])
def message():
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()

    # 사용자 목록 가져오기
    cursor.execute("SELECT studentID, username FROM users")
    users = cursor.fetchall()  # 모든 사용자 정보 가져오기

    # 메시지 목록 가져오기
    cursor.execute(""" 
        SELECT m.sender_id, m.title, m.content, u.username AS sender, m.receiver_id
        FROM messages AS m
        JOIN users AS u ON m.sender_id = u.studentID
        WHERE m.receiver_id = ? OR m.sender_id = ?
    """, (session.get('studentID'), session.get('studentID')))  # 현재 사용자의 ID로 메시지 필터링
    messages = cursor.fetchall()
    print(f"Session studentID: {session.get('studentID')}")

    conn.close()

    return render_template('message.html', users=users)  # HTML만 반환

# 메시지 사용자 정보 라우트
@app.route('/api/messages', methods=['GET'])
def get_messages():
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()

    # 사용자 목록 가져오기
    cursor.execute("SELECT studentID, username FROM users")
    users = cursor.fetchall()  # 모든 사용자 정보 가져오기

    # reply_to_message_id가 NULL인 메시지 목록 가져오기 (메인 메시지만 가져옴)
    cursor.execute(""" 
        SELECT m.message_id, m.sender_id, m.title, m.content, m.receiver_id
        FROM messages AS m
        JOIN users AS u ON m.sender_id = u.studentID
        WHERE (m.receiver_id = ? OR m.sender_id = ?) AND m.reply_to_message_id IS NULL
    """, (session.get('studentID'), session.get('studentID')))  # 현재 사용자의 ID로 메시지 필터링
    messages = cursor.fetchall()

    conn.close()

    # JSON으로 반환
    return jsonify({'messages': messages, 'users': users})

# 메시지 답변 라우트
@app.route('/api/replies/<int:message_id>', methods=['GET'])
def get_replies(message_id):
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()

    # 선택한 message_id와 그에 대한 답변들을 가져오기
    cursor.execute("""
        SELECT m.message_id, m.sender_id, m.title, m.content, m.receiver_id
        FROM messages AS m
        WHERE m.reply_to_message_id = ? OR m.message_id = ?
    """, (message_id, message_id))  # message_id와 reply_to_message_id가 동일한 메시지도 가져오기
    replies = cursor.fetchall()

    conn.close()

    # JSON으로 반환
    return jsonify({'replies': replies})


# 메시지 전송 라우트
@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    receiver_id = data.get('user_id') or data.get('replyReceiverId')  # 받는 사람 ID
    title = data.get('title', '제목 없음')  # 제목이 없을 경우 기본 제목
    content = data['content']
    reply_to_message_id = data.get('reply_to_message_id')  # 답변할 메시지의 ID

    sender_id = session.get('studentID')  # 현재 로그인한 사용자의 ID

    # 자신에게 메시지 보내는지 확인
    if sender_id == receiver_id:
        return jsonify({'status': 'error', 'message': '자신에게 메시지를 보낼 수 없습니다.'}), 400

    print(f"Sending message from {sender_id} to {receiver_id}: {title} - {content}")

    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()

    # 메시지를 데이터베이스에 저장 (reply_to_message_id 포함)
    cursor.execute("""
        INSERT INTO messages (sender_id, receiver_id, title, content, reply_to_message_id) 
        VALUES (?, ?, ?, ?, ?)
    """, (sender_id, receiver_id, title, content, reply_to_message_id))
    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'message': '메시지가 성공적으로 전송되었습니다.'})


@app.route('/base')
def base():
    return render_template('base.html')  # nav바 페이지 렌더링

# 트래킹 시작 라우트 
@app.route('/track_attendance', methods=['POST'])
def track_attendance():
    global track_running
    data = request.get_json()
    class_id = data.get('class_id')
    week = data.get('week')
    
    # 트래킹 시작 플래그를 True로 설정
    track_running = True
    
    # 필요하다면 class_id와 week를 처리하는 로직 추가
    
    # 얼굴 인식 및 트래킹을 시작
    return jsonify({'status': 'tracking started'})


async def client():
    uri = "ws://127.0.0.1:8000"
    async with websockets.connect(uri) as websocket:
        # 서버에 메시지 전송
        message = "Hello, Server!"
        await websocket.send(message)
        print(f"Sent to server: {message}")

        # 서버로부터 응답 받기
        response = await websocket.recv()
        print(f"Received from server: {response}")

# 로그아웃 라우트
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=8000)
    asyncio.run(client())
