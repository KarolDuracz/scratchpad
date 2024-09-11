from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sqlite3
from datetime import datetime

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        self.conn = sqlite3.connect('server_logs.db', check_same_thread=False)
        self.create_table()
        super().__init__(*args, **kwargs)

    def create_table(self):
        # Create the table to store logs
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    request_data TEXT NOT NULL
                )
            ''')
    
    def log_to_db(self, request_data):
        # Insert log entry into the database
        with self.conn:
            self.conn.execute('''
                INSERT INTO logs (time, request_data) 
                VALUES (?, ?)
            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), request_data))

    def fetch_logs_from_db(self):
        # Fetch logs from the database
        with self.conn:
            return self.conn.execute('SELECT * FROM logs').fetchall()

    def do_GET(self):
        if self.path == '/logs':
            # Serve the logs page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            # Fetch logs from the database
            logs = self.fetch_logs_from_db()
            log_html = "<br>".join([f"{log[1]} - {log[2]}" for log in logs])  # Format logs as HTML

            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Logs</title>
            </head>
            <body>
                <h1>Server Logs</h1>
                <div>{log_html}</div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            # Serve the client page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AJAX to Python Server</title>
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            </head>
            <body>
                <h1>Send Signal to Python Server</h1>
                <button id="sendBtn">Send Signal</button>
                <div id="responseMessages"></div>
                <script>
                    $(document).ready(function(){
                        $('#sendBtn').click(function(){
                            const data = JSON.stringify({ 'signal': 'button_clicked' });
                            $.ajax({
                                url: 'http://localhost:8080',  // Python server URL
                                type: 'POST',
                                contentType: 'application/json',  // Sending JSON data
                                data: data,
                                success: function(response) {
                                    $('#responseMessages').append('<p>' + response.message + '</p>');
                                },
                                error: function(jqXHR, textStatus, errorThrown) {
                                    $('#responseMessages').append('<p>Error: ' + textStatus + ' - ' + errorThrown + '</p>');
                                }
                            });
                        });
                    });
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        # Handle POST request
        content_length = int(self.headers['Content-Length'])  # Get the size of the POST data
        post_data = self.rfile.read(content_length).decode('utf-8')  # Read the POST data
        print(f"Received: {post_data}")  # Print the POST data

        # Log request data to the database
        self.log_to_db(post_data)

        # Send response headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # Respond back with JSON
        response = {'status': 'success', 'message': 'Data received successfully!'}
        self.wfile.write(json.dumps(response).encode('utf-8'))

if __name__ == "__main__":
    server_address = ('', 8080)  # Server listens on port 8080
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print("Starting server on port 8080...")
    httpd.serve_forever()
