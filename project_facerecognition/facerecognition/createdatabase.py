import sqlite3

# 통합된 데이터베이스 파일 생성
db_path = r'C:\Users\youji\Downloads\sqlite-tools-win-x64-3470000\university.db'

def create_users():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(100) NOT NULL,
            email VARCHAR(100) NOT NULL,
            password VARCHAR(255) NOT NULL,
            is_admin BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            studentID VARCHAR(50) NOT NULL UNIQUE,
            department VARCHAR(100)
        )
    ''')
    conn.commit()
    conn.close()

def create_departments():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            department_id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_name VARCHAR(100) NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def create_students():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student (
            studentID VARCHAR(50) PRIMARY KEY,
            department_id INTEGER,
            FOREIGN KEY (department_id) REFERENCES departments(department_id)
        )
    ''')
    conn.commit()
    conn.close()

def create_classes():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            class_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name VARCHAR(100) NOT NULL,
            class_code VARCHAR(50) NOT NULL,
            section VARCHAR(50) NOT NULL,
            professor_id INTEGER,
            FOREIGN KEY (professor_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def create_enrollment():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS enrollment (
            enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id VARCHAR(50),
            class_id INTEGER,
            FOREIGN KEY (student_id) REFERENCES student(studentID),
            FOREIGN KEY (class_id) REFERENCES classes(class_id)
        )
    ''')
    conn.commit()
    conn.close()

def create_messages():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER,
            receiver_id INTEGER,
            class_id INTEGER,
            content TEXT,
            is_important BOOLEAN DEFAULT FALSE,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sender_id) REFERENCES users(id),
            FOREIGN KEY (receiver_id) REFERENCES users(id),
            FOREIGN KEY (class_id) REFERENCES classes(class_id)
        );
    ''')
    conn.commit()
    conn.close()

def create_attendance():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id VARCHAR(50),
            class_id INTEGER,
            attendance_date DATE DEFAULT (DATE('now')),
            status VARCHAR(10),
            FOREIGN KEY (student_id) REFERENCES student(studentID),
            FOREIGN KEY (class_id) REFERENCES classes(class_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# 새로운 events 테이블 생성 함수
def create_events():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 기존 events 테이블 삭제 (이미 존재하는 경우)
    cursor.execute('DROP TABLE IF EXISTS events')
    
    # 새로운 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT,
            description TEXT,
            studentID VARCHAR(50) NOT NULL,  -- studentID 필드 추가
            FOREIGN KEY (studentID) REFERENCES users(studentID)  -- users 테이블의 studentID와 연결
        )
    ''')
    conn.commit()
    conn.close()
    print("Events 테이블이 생성되었습니다.")


if __name__ == '__main__':
    # 필요한 모든 테이블 생성
    create_users()
    create_departments()
    create_students()
    create_classes()
    create_enrollment()
    create_messages()
    create_attendance()
    create_events()  # 새로운 events 테이블 생성 호출
