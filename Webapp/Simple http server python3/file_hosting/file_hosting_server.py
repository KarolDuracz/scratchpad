import os
import cgi
import posixpath
import http.server
from urllib.parse import unquote
import html  # Import html module for HTML escaping

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
        displaypath = html.escape(unquote(self.path))  # Fixed line

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        # Start HTML response
        html_response = f"""
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
            html_response += f'<li><a href="{linkname}">{displayname}</a></li>'
        
        html_response += """
            </ul>
        </body>
        </html>
        """

        self.wfile.write(html_response.encode('utf-8'))

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
