from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        # Handle GET request by serving a basic HTML response
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
            <p id="responseMessage"></p>
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
                                $('#responseMessage').html('Server response: ' + response.message);
                            },
                            error: function(jqXHR, textStatus, errorThrown) {
                                $('#responseMessage').html('Error: ' + textStatus + ' - ' + errorThrown);
                            }
                        });
                    });
                });
            </script>
        </body>
        </html>
        """
        # Write the HTML content as response
        self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        # Handle POST request
        content_length = int(self.headers['Content-Length'])  # Get the size of the POST data
        post_data = self.rfile.read(content_length)  # Read the POST data
        print(f"Received: {post_data.decode('utf-8')}")  # Print the POST data

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
