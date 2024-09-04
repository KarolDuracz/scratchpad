run - python page-db3.py <br />
http://localhost:8000/ <br />
http://localhost:8000/list<br />
http://localhost:8000/admin<br />
login: admin1<br />
password: admin<br />

No description. Sorry. Other files modify .db file add columns, roles, user, etc.
<br /><br />
+ track_simulator.html for this http server<br />
This is to demonstrate a simple game and a backend, i.e. and python server.

<b> <h2>But like everything here, there is something started and unfinished .  </h2></b>
Maybe someday...

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Webapp/Simple%20http%20server%20python3/simple%20game.png)

<hr>

First touch

```
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse

# Initialize the SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS submissions (id INTEGER PRIMARY KEY, topic TEXT, text TEXT)''')
conn.commit()

# Define the web server request handler
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # Serve the HTML form on GET request
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
        </body>
        </html>
        '''
        self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        # Parse the form data posted
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = urlparse.parse_qs(post_data.decode('utf-8'))

        # Extract the topic and text
        topic = data.get('topic', [''])[0]
        text = data.get('text', [''])[0]

        # Insert the data into the SQLite database
        c.execute("INSERT INTO submissions (topic, text) VALUES (?, ?)", (topic, text))
        conn.commit()

        # Send a response back to the client
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Thank you! Your submission has been saved.')

# Run the web server
def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Server running on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
```
How it works:
SQLite Database:

A SQLite database named data.db is initialized, with a table called submissions for storing the topic and text.
Web Server:

The server handles two types of requests:
GET: Serves an HTML form to the user.
POST: Processes the form data, saves it to the SQLite database, and responds with a confirmation message.
Form Submission:

The form contains two inputs, topic and text, which the user fills out. When the form is submitted, the data is sent to the server via a POST request.
Running the Server:
Save the code to a file, e.g., server.py.
Run the server with python server.py.
Visit http://localhost:8000 in a web browser to interact with the form.
This code provides a basic framework for handling GET and POST requests and storing form data in a SQLite database using Python.
