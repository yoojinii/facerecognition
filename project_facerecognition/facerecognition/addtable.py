
import sqlite3

# 통합된 데이터베이스 파일 생성
db_path = r'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\university.db'

def add_reply_to_field():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 열이 이미 존재하는지 확인
    cursor.execute("PRAGMA table_info(messages)")
    columns = [column[1] for column in cursor.fetchall()]
    if "reply_to_message_id" not in columns:
        cursor.execute('''
            ALTER TABLE messages
            ADD COLUMN reply_to_message_id INTEGER REFERENCES messages(message_id)
        ''')
        conn.commit()
    conn.close()

add_reply_to_field()
