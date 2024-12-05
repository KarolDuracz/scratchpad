// it doesn't work - this is only example how to do that in some way
// antoher way ffmpeg -i rtsp://<camera-ip>/live output.mp4

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <iostream>
#include <fstream>

// Link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

// Global Variables
const char* camera_ip = "192.168.1.1";  // Camera IP
const char* stream_path = "/video";     // Stream path (change as needed)
const int camera_port = 8080;           // HTTP port for video stream
HWND hwnd;                              // Handle to the main window
HDC hdc;                                // Device context for rendering

//  Create a Window for Display
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

HWND CreateMainWindow(HINSTANCE hInstance) {
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "VideoStreamWindow";

    RegisterClass(&wc);

    return CreateWindowEx(0, wc.lpszClassName, "Camera Stream", WS_OVERLAPPEDWINDOW,
                          CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
                          NULL, NULL, hInstance, NULL);
}

// Establish a Connection and Fetch Stream
SOCKET ConnectToCamera() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Socket creation failed." << std::endl;
        WSACleanup();
        return INVALID_SOCKET;
    }

    sockaddr_in serverAddr = {};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(camera_port);
    inet_pton(AF_INET, camera_ip, &serverAddr.sin_addr);

    if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Connection to camera failed." << std::endl;
        closesocket(sock);
        WSACleanup();
        return INVALID_SOCKET;
    }

    // Send HTTP GET request
    std::string getRequest = "GET " + std::string(stream_path) + " HTTP/1.1\r\nHost: " + camera_ip + "\r\n\r\n";
    send(sock, getRequest.c_str(), getRequest.size(), 0);

    return sock;
}

// Decode and Display Frames (Simplified for MJPEG)
void DisplayStream(SOCKET sock) {
    char buffer[4096];
    int bytesReceived;
    std::ofstream jpegFile;

    while ((bytesReceived = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        // Example: Save to file if a JPEG boundary is detected
        std::string data(buffer, bytesReceived);
        size_t pos = data.find("\xff\xd8");  // JPEG start marker
        if (pos != std::string::npos) {
            // Write to a file for testing
            jpegFile.open("frame.jpg", std::ios::binary);
            jpegFile.write(data.c_str() + pos, bytesReceived - pos);
            jpegFile.close();

            // Render the frame (Simplified for testing)
            InvalidateRect(hwnd, NULL, TRUE);
            HBITMAP hBitmap = (HBITMAP)LoadImage(NULL, "frame.jpg", IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
            HDC memDC = CreateCompatibleDC(hdc);
            SelectObject(memDC, hBitmap);
            BitBlt(hdc, 0, 0, 800, 600, memDC, 0, 0, SRCCOPY);
            DeleteDC(memDC);
        }
    }
}

// Main Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    hwnd = CreateMainWindow(hInstance);
    ShowWindow(hwnd, nCmdShow);

    SOCKET sock = ConnectToCamera();
    if (sock != INVALID_SOCKET) {
        hdc = GetDC(hwnd);
        DisplayStream(sock);
        ReleaseDC(hwnd, hdc);
        closesocket(sock);
    }

    WSACleanup();
    return 0;
}

/* DEMO 2 */

#include <iostream>
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")

void FetchHttpStream(const std::string& cameraIP) {
    WSADATA wsaData;
    SOCKET sock = INVALID_SOCKET;
    struct sockaddr_in server;

    // Initialize Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Failed to initialize Winsock. Error: " << WSAGetLastError() << std::endl;
        return;
    }

    // Create the socket
    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Failed to create socket. Error: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return;
    }

    // Setup server address
    server.sin_family = AF_INET;
    server.sin_port = htons(8080); // Default GoPro port for HTTP stream
    inet_pton(AF_INET, cameraIP.c_str(), &server.sin_addr);

    // Connect to the camera
    if (connect(sock, (struct sockaddr*)&server, sizeof(server)) == SOCKET_ERROR) {
        std::cerr << "Failed to connect to the camera. Error: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        return;
    }

    // Send HTTP GET request
    const std::string request = "GET /live/stream.m3u8 HTTP/1.1\r\n"
                                "Host: " + cameraIP + ":8080\r\n"
                                "Connection: close\r\n\r\n";
    send(sock, request.c_str(), request.length(), 0);

    // Receive and process data
    char buffer[4096];
    int bytesReceived = 0;
    while ((bytesReceived = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        std::cout.write(buffer, bytesReceived); // Handle raw stream data here
    }

    // Cleanup
    closesocket(sock);
    WSACleanup();
}


