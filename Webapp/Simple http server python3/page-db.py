from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse
import sqlite3
from datetime import datetime
import bcrypt
import os
from dotenv import load_dotenv
from http import cookies
import random

# Load environment variables from .env file
load_dotenv()

ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH')

# Setup in-memory session store (simple implementation)
sessions = {}

# Connect to SQLite database
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/admin":
            self.handle_admin_page()
        elif self.path == "/list":
            self.handle_list_page()
        elif self.path == "/login":
            self.handle_login_page()
        else:
            session_id = self.get_session_id()
            if session_id and sessions.get(session_id):
                self.handle_form_page()
            else:
                self.redirect_to_login()

    def do_POST(self):
        if self.path == "/login":
            self.handle_login_submission()
        elif self.path == "/admin":
            session_id = self.get_session_id()
            if session_id and sessions.get(session_id):
                self.handle_admin_submission()
            else:
                self.redirect_to_login()
        else:
            session_id = self.get_session_id()
            if session_id and sessions.get(session_id):
                self.handle_form_submission()
            else:
                self.redirect_to_login()

    # Check if session exists and get session id
    def get_session_id(self):
        if "Cookie" in self.headers:
            cookie_header = self.headers.get('Cookie')
            cookie = cookies.SimpleCookie(cookie_header)
            if "session_id" in cookie:
                return cookie["session_id"].value
        return None

    # Redirect to login page if not authenticated
    def redirect_to_login(self):
        self.send_response(302)
        self.send_header('Location', '/login')
        self.end_headers()

    # Handle login page
    def handle_login_page(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = '''
        <html>
        <body>
            <h1>Login</h1>
            <form action="/login" method="post">
                Username: <input type="text" name="username"><br>
                Password: <input type="password" name="password"><br>
                <input type="submit" value="Login">
            </form>
            <a href="/">Back to Home</a>
        </body>
        </html>
        '''
        self.wfile.write(html.encode('utf-8'))

    # Handle login form submission
    def handle_login_submission(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = urlparse.parse_qs(post_data.decode('utf-8'))

        username = data.get('username', [''])[0]
        password = data.get('password', [''])[0].encode('utf-8')

        if username == ADMIN_USERNAME and bcrypt.checkpw(password, ADMIN_PASSWORD_HASH.encode('utf-8')):
            # Create session
            session_id = str(random.randint(100000, 999999))
            sessions[session_id] = username

            # Set cookie
            self.send_response(302)
            self.send_header('Set-Cookie', f'session_id={session_id}')
            self.send_header('Location', '/')
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Login failed! <a href="/login">Try again</a>')

    # Handle form submission and save data to the database
    def handle_form_submission(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = urlparse.parse_qs(post_data.decode('utf-8'))

            topic = data.get('topic', [''])[0]
            text = data.get('text', [''])[0]
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            c.execute("INSERT INTO submissions (topic, text, date) VALUES (?, ?, ?)", (topic, text, date))
            conn.commit()

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Thank you! Your submission has been saved.')

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            error_message = f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
            self.wfile.write(error_message.encode('utf-8'))

    # Handle admin page
    def handle_admin_page(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = '''
        <html>
        <body>
            <h1>Admin Page</h1>
            <form action="/admin" method="post">
                New Username: <input type="text" name="new_username"><br>
                New Password: <input type="password" name="new_password"><br>
                <input type="submit" value="Create User">
            </form>
            <a href="/">Back to Home</a>
        </body>
        </html>
        '''
        self.wfile.write(html.encode('utf-8'))

    # Handle admin form submission to create new users
    def handle_admin_submission(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = urlparse.parse_qs(post_data.decode('utf-8'))

        new_username = data.get('new_username', [''])[0]
        new_password = data.get('new_password', [''])[0].encode('utf-8')

        # Hash the new user's password before storing it
        hashed_password = bcrypt.hashpw(new_password, bcrypt.gensalt())

        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_username, hashed_password.decode('utf-8')))
            conn.commit()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'User created successfully! <a href="/admin">Back to Admin Page</a>')
        except sqlite3.IntegrityError:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Username already exists! <a href="/admin">Try again</a>')

    # Display all topics and texts stored in the database
    def handle_list_page(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        c.execute("SELECT topic, text, date FROM submissions")
        rows = c.fetchall()

        html = '''
        <html>
        <body>
            <h1>List of Topics and Texts</h1>
            <ul>
        '''

        for row in rows:
            html += f"<li><strong>Topic:</strong> {row[0]}<br><strong>Text:</strong> {row[1]}<br><strong>Date:</strong> {row[2]}</li><br>"

        html += '''
            </ul>
            <a href="/">Back to Home</a>
        </body>
        </html>
        '''
        self.wfile.write(html.encode('utf-8'))

    # HTML form for submitting topic and text
    def handle_form_page(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = '''
        <html>
        <body>
            <h1>Submit your topic and text</h1>
            <form action="/" method="post">
                Topic: <input type="text" name="topic"><br>
                Text: <textarea name="text"></textarea><br>
                <input type="submit" value="Submit">
            </form>
            <a href="/login">Login</a> | <a href="/list">List Topics</a>
        </body>
        </html>
        '''
        self.wfile.write(html.encode('utf-8'))

# Run the web server
def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Server running on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
