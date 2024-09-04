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
        session_id = self.get_session_id()
        if self.path == "/admin":
            if session_id and sessions.get(session_id):
                self.handle_admin_page()
            else:
                self.redirect_to_login()
        elif self.path == "/list":
            if session_id and sessions.get(session_id):
                self.handle_list_page()
            else:
                self.redirect_to_login()
        elif self.path.startswith("/edit_user"):
            if session_id and sessions.get(session_id):
                self.handle_edit_user_page()
            else:
                self.redirect_to_login()
        elif self.path.startswith("/delete_user"):
            if session_id and sessions.get(session_id):
                self.handle_delete_user_page()
            else:
                self.redirect_to_login()
        elif self.path == "/login":
            self.handle_login_page()
        elif self.path == "/logout":
            self.handle_logout()
        elif self.path == "/welcome":
            self.handle_welcome_page()
        
        else:
            if session_id and sessions.get(session_id):
                self.handle_form_page()
            else:
                self.redirect_to_login()

    def do_POST(self):
        session_id = self.get_session_id()
        if self.path == "/login":
            self.handle_login_submission()
        elif self.path == "/admin":
            if session_id and sessions.get(session_id):
                self.handle_admin_submission()
            else:
                self.redirect_to_login()
        elif self.path.startswith("/edit_user"):
            if session_id and sessions.get(session_id):
                self.handle_edit_user_submission()
            else:
                self.redirect_to_login()
        elif self.path.startswith("/delete_user"):
            if session_id and sessions.get(session_id):
                self.handle_delete_user_submission()
            else:
                self.redirect_to_login()
        else:
            if session_id and sessions.get(session_id):
                self.handle_form_submission()
            else:
                self.redirect_to_login()

    def get_session_id(self):
        if "Cookie" in self.headers:
            cookie_header = self.headers.get('Cookie')
            cookie = cookies.SimpleCookie(cookie_header)
            if "session_id" in cookie:
                return cookie["session_id"].value
        return None

    def redirect_to_login(self):
        self.send_response(302)
        self.send_header('Location', '/login')
        self.end_headers()

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
        
    def handle_welcome_page(self):
        session_id = self.get_session_id()
        if session_id in sessions:
            username = sessions[session_id]
            c.execute("SELECT role FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            print(result, username, session_id)
    
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''
           <html>
            <body>
                <h1>WELCOME</h1>
                
                <a href="/">Back to Home</a>
            </body>
            </html>
            '''
            self.wfile.write(html.encode('utf-8'))
            
        else:
            # No valid session
            self.send_response(401)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Please log in to access this page.')

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
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = c.fetchone()

            if result and bcrypt.checkpw(password, result[0].encode('utf-8')):
                # Password is correct, handle successful login
                print(" scenario 1 " )
                # Create a new session ID
                session_id = str(random.randint(100000, 999999))

                # Store the session ID and associated username
                sessions[session_id] = username

                # Set the session cookie
                self.send_response(302)
                self.send_header('Set-Cookie', f'session_id={session_id}')
                self.send_header('Location', '/welcome')  # Redirect to the welcome page
                self.end_headers()
            else:
                # Handle login failure
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'Login failed! <a href="/login">Try again</a>')
                print(" scenario 2 " )

    def handle_logout(self):
        session_id = self.get_session_id()
        if session_id in sessions:
            del sessions[session_id]

        self.send_response(302)
        self.send_header('Set-Cookie', 'session_id=; expires=Thu, 01 Jan 1970 00:00:00 GMT')
        self.send_header('Location', '/login')
        self.end_headers()

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

    def handle_admin_page(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Fetch all users from the database
        c.execute("SELECT id, username FROM users")
        users = c.fetchall()

        html = '''
        <html>
        <body>
            <h1>Admin Page</h1>
            <form action="/admin" method="post">
                New Username: <input type="text" name="new_username"><br>
                New Password: <input type="password" name="new_password"><br>
                <input type="submit" value="Create User">
            </form>
            <h2>Existing Users</h2>
            <ul>
        '''

        for user in users:
            user_id, username = user
            html += f'''
            <li>
                <strong>{username}</strong>
                <a href="/edit_user?id={user_id}">Edit</a> |
                <a href="/delete_user?id={user_id}">Delete</a>
            </li>
            '''

        html += '''
            </ul>
            <a href="/">Back to Home</a> | <a href="/logout">Logout</a>
        </body>
        </html>
        '''
        self.wfile.write(html.encode('utf-8'))

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

    def handle_edit_user_page(self):
        user_id = self.get_query_param('id')

        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()

        if user:
            username = user[0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f'''
            <html>
            <body>
                <h1>Edit User</h1>
                <form action="/edit_user?id={user_id}" method="post">
                    Username: <input type="text" name="username" value="{username}"><br>
                    New Password: <input type="password" name="new_password"><br>
                    New Role: <input type="role" name="new_role" value="{username}"><br>
                    <input type="submit" value="Update User">
                </form>
                <a href="/admin">Back to Admin Page</a>
            </body>
            </html>
            '''
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'User not found.')

    def handle_edit_user_submission(self):
        user_id = self.get_query_param('id')
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = urlparse.parse_qs(post_data.decode('utf-8'))

        new_username = data.get('username', [''])[0]
        new_password = data.get('new_password', [''])[0].encode('utf-8')
        
        print(" ----> ", new_password)

        hashed_password = bcrypt.hashpw(new_password, bcrypt.gensalt()) if new_password else None

        if hashed_password:
            c.execute("UPDATE users SET username = ?, password = ? WHERE id = ?", (new_username, hashed_password.decode('utf-8'), user_id))
        else:
            c.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, user_id))

        conn.commit()

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'User updated successfully! <a href="/admin">Back to Admin Page</a>')

    def handle_delete_user_page(self):
        user_id = self.get_query_param('id')

        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()

        if user:
            username = user[0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f'''
            <html>
            <body>
                <h1>Delete User</h1>
                <p>Are you sure you want to delete user "{username}"?</p>
                <form action="/delete_user?id={user_id}" method="post">
                    <input type="submit" value="Delete">
                </form>
                <a href="/admin">Back to Admin Page</a>
            </body>
            </html>
            '''
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'User not found.')

    def handle_delete_user_submission(self):
        user_id = self.get_query_param('id')

        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'User deleted successfully! <a href="/admin">Back to Admin Page</a>')

    def get_query_param(self, param):
        parsed_path = urlparse.urlparse(self.path)
        query_params = urlparse.parse_qs(parsed_path.query)
        return query_params.get(param, [None])[0]
        
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
            <a href="/">Back to Home</a> | <a href="/logout">Logout</a>
        </body>
        </html>
        '''
        self.wfile.write(html.encode('utf-8'))

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
            <a href="/logout">Logout</a> | <a href="/list">List Topics</a>
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
