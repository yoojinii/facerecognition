import sqlite3
db_path = 'C:\Users\youji\OneDrive\project_facerecognition\project_facerecognition\facerecognition\university.db'
def clear_attendance_records():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM attendance')
    #cursor.execute('DELETE FROM department')
    conn.commit()
    conn.close()

clear_attendance_records()