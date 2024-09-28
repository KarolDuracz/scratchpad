...

<hr>

This is an old implementation. This is similar to .py file but this used 
```cgi.escape(unquote(self.path))``` 
and generate error ```AttributeError: module 'cgi' has no attribute 'escape'```
on my current installation in Windows system 
```cmd> python --version // Python 3.12.2``` 
But I want to keep here this code. This is an old implementation probably Python 3.2 and older. And removed in Python 3.8. And we need change to ```displaypath = html.escape(unquote(self.path))```


```
import os
import cgi
import posixpath
import http.server
from urllib.parse import unquote

class SimpleFileHostingHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle file upload"""
        content_type = self.headers.get('Content-Type')
        if not content_type.startswith('multipart/form-data'):
            self.send_error(400, "Bad Request: Expected multipart/form-data")
            return
        
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )

        # Check if there's a file in the form data
        if 'file' not in form:
            self.send_error(400, "Bad Request: No file provided")
            return

        # Get file item
        file_item = form['file']
        if not file_item.filename:
            self.send_error(400, "Bad Request: No filename provided")
            return

        filename = os.path.basename(file_item.filename)
        upload_dir = 'uploads'

        # Create upload directory if not exists
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Save file to the uploads directory
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, 'wb') as output_file:
            output_file.write(file_item.file.read())

        # Respond with success
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"File uploaded successfully!")

    def list_directory(self, path):
        """Display file list with a file upload form."""
        try:
            list_dir = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None
        
        list_dir.sort(key=lambda a: a.lower())
        displaypath = cgi.escape(unquote(self.path))

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        # Start HTML response
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>File Hosting</title>
        </head>
        <body>
            <h2>Upload a file</h2>
            <form enctype="multipart/form-data" method="post">
                <input name="file" type="file"/>
                <input type="submit" value="Upload"/>
            </form>
            <hr>
            <h2>Directory listing for {displaypath}</h2>
            <ul>
        """

        # Add file listing
        for name in list_dir:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            html += f'<li><a href="{linkname}">{displayname}</a></li>'
        
        html += """
            </ul>
        </body>
        </html>
        """

        self.wfile.write(html.encode('utf-8'))

if __name__ == '__main__':
    import argparse
    import socketserver

    parser = argparse.ArgumentParser(description="Simple File Hosting Server")
    parser.add_argument('--port', type=int, default=8000, help='Specify alternate port [default: 8000]')
    args = parser.parse_args()

    Handler = SimpleFileHostingHandler
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"Serving on port {args.port}")
        httpd.serve_forever()
```
I don't know which this command is correct butprobably not. To run this I need ```python file_hosting_server.py --port 8000```

```
python -m http.server --directory /path/to/files 8000
```
