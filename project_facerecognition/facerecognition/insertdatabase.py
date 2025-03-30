import sqlite3

department_data = [
    ('정보통신학과',),
    ('정보보호학과',)
]
conn = sqlite3.connect('C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\university.db')
cursor = conn.cursor()
cursor.executemany('''
                   INSERT INTO departments (department_name)
                   VALUES (?)
                   ''', department_data)

conn.commit()
conn.close()

classes_data = [
    (1, 'C언어', 'C1', 3, '001'),
        (2, '시스템보안프로젝트', 'C2', 5, '002'),
        (3, '데이터베이스', 'C3', 4, '001'),
        (4, '통신이론', 'C4', 2, '001'),
        (5, '전공진로세미나3', 'C5', 1, '003'),
        (6, '운영체제', 'C6', 3, '001'),
        (7, '알고리즘', 'C7', 2, '002'),
        (8, '컴퓨터네트워크', 'C8', 4, '001')
]

conn = sqlite3.connect('C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\university.db')
cursor = conn.cursor()
cursor.executemany('''
                INSERT INTO classes (class_id, class_name, class_code, professor_id, section)
                VALUES (?, ?, ?, ?, ?)
                ''', classes_data)
conn.commit()
conn.close()

users_data = [
    ('교수님1','pro1@naver.com','1234','21000001','정보보호학과',True),
    ('교수님2','pro2@naver.com','1234','21000002','정보보호학과',True),
    ('교수님3','pro3@naver.com','1234','21000003','정보보호학과',True),
    ('교수님4','pro4@naver.com','1234','21000004','정보보호학과',True),
    ('교수님5','pro5@naver.com','1234','21000005','정보보호학과',True),
    ('김우경','wk@naver.com','1234','21018017','정보보호학과',False),
    ('국소희','sh@naver.com','1234','21018004','정보보호학과',False),
    ('안유진','yj@naver.com','1234','21018049','정보보호학과',False),
    ('김인서','is@naver.com','1234','21018018','정보통신학과',False),
    ('김채린','cr@naver.com','1234','21018022','정보보호학과',False),
]
conn = sqlite3.connect('C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\university.db')
cursor = conn.cursor()
cursor.executemany('''
                INSERT INTO users (username, email, password,studentID,department,is_admin)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', users_data)
conn.commit()
conn.close()


student_data = [
    ('21018017','2'),
    ('21018004','2'),
    ('21018049','2'),
    ('21018018','1'),
    ('21018022','2'),
]
conn = sqlite3.connect('C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\university.db')
cursor = conn.cursor()
cursor.executemany('''
                   INSERT INTO student (studentID,department_id)
                   VALUES (?,?)
                   ''', student_data)

conn.commit()
conn.close()

enrollment_data = [
    ('21018017','1'),
    ('21018017','7'),
    ('21018017','8'),
    ('21018004','1'),
    ('21018004','2'),
    ('21018004','7'),
    ('21018049','1'),
    ('21018049','5'),
    ('21018049','3'),
    ('21018018','1'),
    ('21018018','4'),
    ('21018018','5'),
    ('21018022','1'),
    ('21018022','2'),
    ('21018022','8'),
]
conn = sqlite3.connect('C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\university.db')
cursor = conn.cursor()
cursor.executemany('''
                INSERT INTO enrollment (student_id,class_id)
                VALUES (?,?)
                ''', enrollment_data)
conn.commit()
conn.close()
###아직 실행 x

