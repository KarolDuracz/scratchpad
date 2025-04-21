from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)

DATABASE = 'database.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # OPTIONAL: Drop table if it exists to force recreate (if you don't need existing data)
    # RESET DB
    #cursor.execute('DROP TABLE IF EXISTS tasks')

    # Recreate table with updated CHECK constraint
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            hour TEXT,
            topic TEXT NOT NULL,
            text TEXT,
            status TEXT NOT NULL CHECK(status IN ('planned', 'done', 'note'))
        )
    ''')
    conn.commit()
    conn.close()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks():
    if request.method == 'GET':
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, date, hour, topic, text, status FROM tasks")
        tasks = cursor.fetchall()
        conn.close()
        return jsonify(tasks)
    elif request.method == 'POST':
        data = request.get_json()
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tasks (date, hour, topic, text, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['date'], data.get('hour'), data['topic'], data.get('text'), data['status']))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Task added successfully'}), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    data = request.get_json()
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE tasks
        SET date = ?, hour = ?, topic = ?, text = ?, status = ?
        WHERE id = ?
    ''', (data['date'], data['hour'], data['topic'], data['text'], data['status'], task_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Task updated successfully'})

@app.route('/api/top-tasks')
def top_tasks():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, date, hour, topic, text, status
        FROM tasks
        WHERE status != 'done'
        ORDER BY date ASC, hour ASC
        LIMIT 10
    ''')
    tasks = cursor.fetchall()
    conn.close()
    return jsonify(tasks)


if __name__ == '__main__':
    init_db()
    # app.run(debug=True) # onyl localhost 
    app.run(debug=True, host='0.0.0.0')

