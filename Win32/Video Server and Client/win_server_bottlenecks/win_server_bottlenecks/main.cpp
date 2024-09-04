#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <wincrypt.h>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <iomanip>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Advapi32.lib")  // For CryptoAPI

#define FRAME_SIZE 4096  // Simulated frame size
#define PORT 8082

std::queue<std::vector<char>> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCV;

// Base64 encoding function
std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            ret += '=';
    }

    return ret;
}

// Function to generate the Sec-WebSocket-Accept key
std::string generate_websocket_accept_key(const std::string& key) {
    std::string magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string accept_key = key + magic_string;

    HCRYPTPROV hProv = 0;
    HCRYPTHASH hHash = 0;
    BYTE hash[20];
    DWORD hashLen = 20;

    // Acquire a cryptographic provider context handle.
    if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
        std::cerr << "CryptAcquireContext failed." << std::endl;
        return "";
    }

    // Create a hash object.
    if (!CryptCreateHash(hProv, CALG_SHA1, 0, 0, &hHash)) {
        std::cerr << "CryptCreateHash failed." << std::endl;
        CryptReleaseContext(hProv, 0);
        return "";
    }

    // Hash the data.
    if (!CryptHashData(hHash, reinterpret_cast<const BYTE*>(accept_key.c_str()), accept_key.length(), 0)) {
        std::cerr << "CryptHashData failed." << std::endl;
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        return "";
    }

    // Retrieve the hash value.
    if (!CryptGetHashParam(hHash, HP_HASHVAL, hash, &hashLen, 0)) {
        std::cerr << "CryptGetHashParam failed." << std::endl;
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        return "";
    }

    // Clean up.
    CryptDestroyHash(hHash);
    CryptReleaseContext(hProv, 0);

    // Base64 encode the hash result
    return base64_encode(hash, hashLen);
}

void handleClient(SOCKET clientSocket) {
    char recvbuf[FRAME_SIZE];
    int bytesReceived;

    // Handle WebSocket handshake
    bytesReceived = recv(clientSocket, recvbuf, FRAME_SIZE, 0);
    if (bytesReceived > 0) {
        std::string request(recvbuf, bytesReceived);
        std::size_t keyPos = request.find("Sec-WebSocket-Key: ");
        if (keyPos != std::string::npos) {
            keyPos += 19;
            std::size_t keyEnd = request.find("\r\n", keyPos);
            std::string secWebSocketKey = request.substr(keyPos, keyEnd - keyPos);

            std::string secWebSocketAccept = generate_websocket_accept_key(secWebSocketKey);
            std::string response =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " + secWebSocketAccept + "\r\n\r\n";

            send(clientSocket, response.c_str(), response.size(), 0);

            std::cout << "WebSocket connection established with client." << std::endl;
        }
    }
    else {
        std::cerr << "Failed to receive WebSocket handshake from client." << std::endl;
        closesocket(clientSocket);
        return;
    }

    // Send frames to the client
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [] { return !frameQueue.empty(); });

        auto frameData = frameQueue.front();
        frameQueue.pop();

        lock.unlock();

        // Simplified WebSocket frame
        std::vector<char> wsFrame;
        wsFrame.push_back(0x82);  // Binary frame, FIN bit set
        if (frameData.size() <= 125) {
            wsFrame.push_back(static_cast<char>(frameData.size()));
        }
        else if (frameData.size() <= 65535) {
            wsFrame.push_back(126);
            wsFrame.push_back((frameData.size() >> 8) & 0xFF);
            wsFrame.push_back(frameData.size() & 0xFF);
        }
        else {
            wsFrame.push_back(127);
            for (int i = 7; i >= 0; --i) {
                wsFrame.push_back((frameData.size() >> (8 * i)) & 0xFF);
            }
        }

        wsFrame.insert(wsFrame.end(), frameData.begin(), frameData.end());
        send(clientSocket, wsFrame.data(), wsFrame.size(), 0);
    }

    closesocket(clientSocket);
}

void webSocketServer() {
    // WebSocket server for streaming video frames to browser
    SOCKET wsSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in wsAddr;
    wsAddr.sin_family = AF_INET;
    wsAddr.sin_port = htons(PORT);
    wsAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(wsSocket, (SOCKADDR*)&wsAddr, sizeof(wsAddr)) == SOCKET_ERROR) {
        std::cerr << "Bind failed with error: " << WSAGetLastError() << std::endl;
        closesocket(wsSocket);
        WSACleanup();
        return;
    }

    if (listen(wsSocket, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "Listen failed with error: " << WSAGetLastError() << std::endl;
        closesocket(wsSocket);
        WSACleanup();
        return;
    }

    std::cout << "WebSocket server is running on port " << PORT << "..." << std::endl;

    SOCKET clientSocket;
    sockaddr_in clientAddr;
    int clientAddrSize = sizeof(clientAddr);

    while ((clientSocket = accept(wsSocket, (SOCKADDR*)&clientAddr, &clientAddrSize)) != INVALID_SOCKET) {
        std::thread(handleClient, clientSocket).detach();
    }

    closesocket(wsSocket);
}

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    std::thread(webSocketServer).detach();

    // Simulate video frames being captured and queued
    while (true) {
        std::vector<char> frameData(FRAME_SIZE, 0);

        // Simulate capturing frame (fill with random data for now)
        std::generate(frameData.begin(), frameData.end(), []() { return static_cast<char>(rand() % 256); });

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(frameData);
        }
        queueCV.notify_one();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    WSACleanup();
    return 0;
}



// ta implementacja ma zmienione SHA1 i chyba nie dziala przez to
#if 0
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <random>
#include <sstream>
#include <wincrypt.h> //#include <openssl/sha.h>
#include <iomanip>
#include <array>

#pragma comment(lib, "Ws2_32.lib")

#define FRAME_SIZE 4096  // Simulated frame size
#define PORT 8080

std::queue<std::vector<char>> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCV;

// Base64 encoding function
std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            ret += '=';
    }

    return ret;
}

std::string generate_websocket_accept_key(const std::string& key) {
    std::string magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string accept_key = key + magic_string;

    HCRYPTPROV hProv = 0;
    HCRYPTHASH hHash = 0;
    BYTE hash[20];
    DWORD hashLen = 20;

    // Acquire a cryptographic provider context handle.
    if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
        std::cerr << "CryptAcquireContext failed." << std::endl;
        return "";
    }

    // Create a hash object.
    if (!CryptCreateHash(hProv, CALG_SHA1, 0, 0, &hHash)) {
        std::cerr << "CryptCreateHash failed." << std::endl;
        CryptReleaseContext(hProv, 0);
        return "";
    }

    // Hash the data.
    if (!CryptHashData(hHash, reinterpret_cast<const BYTE*>(accept_key.c_str()), accept_key.length(), 0)) {
        std::cerr << "CryptHashData failed." << std::endl;
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        return "";
    }

    // Retrieve the hash value.
    if (!CryptGetHashParam(hHash, HP_HASHVAL, hash, &hashLen, 0)) {
        std::cerr << "CryptGetHashParam failed." << std::endl;
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        return "";
    }

    // Clean up.
    CryptDestroyHash(hHash);
    CryptReleaseContext(hProv, 0);

    // Base64 encode the hash result
    return base64_encode(hash, hashLen);
}


/*
// Function to generate the Sec-WebSocket-Accept key
std::string generate_websocket_accept_key(const std::string& key) {
    std::string magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string accept_key = key + magic_string;

    // SHA-1 hash
    unsigned char hash[20];
    SHA1(reinterpret_cast<const unsigned char*>(accept_key.c_str()), accept_key.length(), hash);

    // Base64 encode
    return base64_encode(hash, sizeof(hash));
}
*/

void handleClient(SOCKET clientSocket) {
    char recvbuf[FRAME_SIZE];
    int bytesReceived;

    // Handle WebSocket handshake
    bytesReceived = recv(clientSocket, recvbuf, FRAME_SIZE, 0);
    if (bytesReceived > 0) {
        std::string request(recvbuf, bytesReceived);
        std::size_t keyPos = request.find("Sec-WebSocket-Key: ");
        if (keyPos != std::string::npos) {
            keyPos += 19;
            std::size_t keyEnd = request.find("\r\n", keyPos);
            std::string secWebSocketKey = request.substr(keyPos, keyEnd - keyPos);

            std::string secWebSocketAccept = generate_websocket_accept_key(secWebSocketKey);
            std::string response =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " + secWebSocketAccept + "\r\n\r\n";

            send(clientSocket, response.c_str(), response.size(), 0);
        }
    }

    // Send frames to the client
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [] { return !frameQueue.empty(); });

        auto frameData = frameQueue.front();
        frameQueue.pop();

        lock.unlock();

        // Simplified WebSocket frame
        std::vector<char> wsFrame;
        wsFrame.push_back(0x82);  // Binary frame, FIN bit set
        if (frameData.size() <= 125) {
            wsFrame.push_back(static_cast<char>(frameData.size()));
        }
        else if (frameData.size() <= 65535) {
            wsFrame.push_back(126);
            wsFrame.push_back((frameData.size() >> 8) & 0xFF);
            wsFrame.push_back(frameData.size() & 0xFF);
        }
        else {
            wsFrame.push_back(127);
            for (int i = 7; i >= 0; --i) {
                wsFrame.push_back((frameData.size() >> (8 * i)) & 0xFF);
            }
        }

        wsFrame.insert(wsFrame.end(), frameData.begin(), frameData.end());
        send(clientSocket, wsFrame.data(), wsFrame.size(), 0);
    }

    closesocket(clientSocket);
}

void httpServer() {
    // Simple HTTP server to serve HTML page
    SOCKET httpSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in httpAddr;
    httpAddr.sin_family = AF_INET;
    httpAddr.sin_port = htons(8081);
    httpAddr.sin_addr.s_addr = INADDR_ANY;

    bind(httpSocket, (SOCKADDR*)&httpAddr, sizeof(httpAddr));
    listen(httpSocket, SOMAXCONN);

    std::cout << "HTTP server is running on port 8081..." << std::endl;

    SOCKET clientSocket;
    sockaddr_in clientAddr;
    int clientAddrSize = sizeof(clientAddr);

    while ((clientSocket = accept(httpSocket, (SOCKADDR*)&clientAddr, &clientAddrSize)) != INVALID_SOCKET) {
        std::string response =
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            "<!DOCTYPE html>"
            "<html>"
            "<body>"
            "<h1>Live Video Stream</h1>"
            "<canvas id='videoCanvas' width='640' height='480'></canvas>"
            "<script>"
            "var ws = new WebSocket('ws://localhost:8082');"
            "ws.binaryType = 'arraybuffer';"
            "ws.onmessage = function(event) {"
            "   var arrayBuffer = event.data;"
            "   var ctx = document.getElementById('videoCanvas').getContext('2d');"
            "   var img = new Image();"
            "   img.onload = function() {"
            "       ctx.drawImage(img, 0, 0);"
            "   };"
            "   var blob = new Blob([new Uint8Array(arrayBuffer)], {type: 'image/jpeg'});"
            "   img.src = URL.createObjectURL(blob);"
            "};"
            "</script>"
            "</body>"
            "</html>";

        send(clientSocket, response.c_str(), response.length(), 0);
        closesocket(clientSocket);
    }
}

void webSocketServer() {
    // WebSocket server for streaming video frames to browser
    SOCKET wsSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in wsAddr;
    wsAddr.sin_family = AF_INET;
    wsAddr.sin_port = htons(8082);
    wsAddr.sin_addr.s_addr = INADDR_ANY;

    bind(wsSocket, (SOCKADDR*)&wsAddr, sizeof(wsAddr));
    listen(wsSocket, SOMAXCONN);

    std::cout << "WebSocket server is running on port 8082..." << std::endl;

    SOCKET clientSocket;
    sockaddr_in clientAddr;
    int clientAddrSize = sizeof(clientAddr);

    while ((clientSocket = accept(wsSocket, (SOCKADDR*)&clientAddr, &clientAddrSize)) != INVALID_SOCKET) {
        std::thread(handleClient, clientSocket).detach();
    }
}

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    // Start the HTTP server to serve the HTML page
    std::thread(httpServer).detach();

    // Start the WebSocket server for video streaming
    std::thread(webSocketServer).detach();

    // Simulate video frames being captured and queued
    while (true) {
        std::vector<char> frameData(FRAME_SIZE, 0);

        // Simulate capturing frame (fill with random data for now)
        std::generate(frameData.begin(), frameData.end(), []() { return static_cast<char>(rand() % 256); });

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(frameData);
        }
        queueCV.notify_one();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    WSACleanup();
    return 0;
}
#endif



#if 0
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>

#pragma comment(lib, "Ws2_32.lib")

#define FRAME_SIZE 4096  // Assume each "frame" is 4 KB of data

int main() {
    // Initialize Winsock
    WSADATA wsaData;
    int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        std::cerr << "WSAStartup failed: " << iResult << std::endl;
        return 1;
    }

    // Create a socket
    SOCKET listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // Set up the sockaddr_in structure
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;  // Bind to any available interface
    serverAddr.sin_port = htons(8080);  // Port 8080

    // Bind the socket
    iResult = bind(listenSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (iResult == SOCKET_ERROR) {
        std::cerr << "Bind failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    // Listen for incoming connections
    iResult = listen(listenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        std::cerr << "Listen failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    std::cout << "Video server is listening on port 8080..." << std::endl;

    SOCKET clientSocket;
    sockaddr_in clientAddr;
    int clientAddrSize = sizeof(clientAddr);

    while (true) {
        clientSocket = accept(listenSocket, (SOCKADDR*)&clientAddr, &clientAddrSize);
        if (clientSocket == INVALID_SOCKET) {
            std::cerr << "Accept failed: " << WSAGetLastError() << std::endl;
            closesocket(listenSocket);
            WSACleanup();
            return 1;
        }

        std::cout << "Client connected..." << std::endl;

        char recvbuf[FRAME_SIZE];
        int bytesReceived;

        // Receive video frames (simulate a stream of frames)
        while ((bytesReceived = recv(clientSocket, recvbuf, FRAME_SIZE, 0)) > 0) {
            std::cout << "Received frame of size: " << bytesReceived << " bytes" << std::endl;
            // Here you can process the "video frame" data
        }

        if (bytesReceived == 0) {
            std::cout << "Connection closing..." << std::endl;
        }
        else if (bytesReceived < 0) {
            std::cerr << "recv failed: " << WSAGetLastError() << std::endl;
        }

        closesocket(clientSocket);
    }

    // Cleanup
    closesocket(listenSocket);
    WSACleanup();
    return 0;
}
#endif 


#if 0
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <fstream>

#pragma comment(lib, "Ws2_32.lib")

#define FRAME_SIZE 4096  // Simulated frame size
#define PORT 8080

std::queue<std::vector<char>> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCV;

void handleClient(SOCKET clientSocket) {
    char recvbuf[FRAME_SIZE];
    int bytesReceived;

    while ((bytesReceived = recv(clientSocket, recvbuf, FRAME_SIZE, 0)) > 0) {
        std::vector<char> frameData(recvbuf, recvbuf + bytesReceived);
        std::unique_lock<std::mutex> lock(queueMutex);
        frameQueue.push(frameData);
        queueCV.notify_one();
    }

    if (bytesReceived == 0) {
        std::cout << "Connection closing..." << std::endl;
    }
    else if (bytesReceived < 0) {
        std::cerr << "recv failed: " << WSAGetLastError() << std::endl;
    }

    closesocket(clientSocket);
}

void httpServer() {
    // Simple HTTP server to serve HTML page
    SOCKET httpSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in httpAddr;
    httpAddr.sin_family = AF_INET;
    httpAddr.sin_port = htons(8081);
    httpAddr.sin_addr.s_addr = INADDR_ANY;

    bind(httpSocket, (SOCKADDR*)&httpAddr, sizeof(httpAddr));
    listen(httpSocket, SOMAXCONN);

    std::cout << "HTTP server is running on port 8081..." << std::endl;

    SOCKET clientSocket;
    sockaddr_in clientAddr;
    int clientAddrSize = sizeof(clientAddr);

    while ((clientSocket = accept(httpSocket, (SOCKADDR*)&clientAddr, &clientAddrSize)) != INVALID_SOCKET) {
        std::string response =
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            "<!DOCTYPE html>"
            "<html>"
            "<body>"
            "<h1>Live Video Stream</h1>"
            "<canvas id='videoCanvas' width='640' height='480'></canvas>"
            "<script>"
            "var ws = new WebSocket('ws://localhost:8082');"
            "ws.binaryType = 'arraybuffer';"
            "ws.onmessage = function(event) {"
            "   var arrayBuffer = event.data;"
            "   var ctx = document.getElementById('videoCanvas').getContext('2d');"
            "   var img = new Image();"
            "   img.onload = function() {"
            "       ctx.drawImage(img, 0, 0);"
            "   };"
            "   var blob = new Blob([new Uint8Array(arrayBuffer)], {type: 'image/jpeg'});"
            "   img.src = URL.createObjectURL(blob);"
            "};"
            "</script>"
            "</body>"
            "</html>";

        send(clientSocket, response.c_str(), response.length(), 0);
        closesocket(clientSocket);
    }
}

void webSocketServer() {
    // WebSocket server for streaming video frames to browser
    SOCKET wsSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in wsAddr;
    wsAddr.sin_family = AF_INET;
    wsAddr.sin_port = htons(8082);
    wsAddr.sin_addr.s_addr = INADDR_ANY;

    bind(wsSocket, (SOCKADDR*)&wsAddr, sizeof(wsAddr));
    listen(wsSocket, SOMAXCONN);

    std::cout << "WebSocket server is running on port 8082..." << std::endl;

    SOCKET clientSocket;
    sockaddr_in clientAddr;
    int clientAddrSize = sizeof(clientAddr);

    while ((clientSocket = accept(wsSocket, (SOCKADDR*)&clientAddr, &clientAddrSize)) != INVALID_SOCKET) {
        std::thread([clientSocket]() {
            std::vector<char> wsHandshakeResponse(
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                "\r\n"
                , std::end("HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    "Sec-WebSocket-Accept: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                    "\r\n")
            );
            send(clientSocket, wsHandshakeResponse.data(), wsHandshakeResponse.size(), 0);

            while (true) {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [] { return !frameQueue.empty(); });

                auto frameData = frameQueue.front();
                frameQueue.pop();

                lock.unlock();

                // Send frame data to the browser (simplified, no WebSocket framing)
                send(clientSocket, frameData.data(), frameData.size(), 0);
            }

            closesocket(clientSocket);
            }).detach();
    }
}

int main() {
    WSADATA wsaData;
    int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        std::cerr << "WSAStartup failed: " << iResult << std::endl;
        return 1;
    }

    SOCKET listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    bind(listenSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    listen(listenSocket, SOMAXCONN);

    std::cout << "Video server is listening on port 8080..." << std::endl;

    std::thread(httpServer).detach();
    std::thread(webSocketServer).detach();

    while (true) {
        SOCKET clientSocket;
        sockaddr_in clientAddr;
        int clientAddrSize = sizeof(clientAddr);

        clientSocket = accept(listenSocket, (SOCKADDR*)&clientAddr, &clientAddrSize);
        if (clientSocket == INVALID_SOCKET) {
            std::cerr << "Accept failed: " << WSAGetLastError() << std::endl;
            continue;
        }

        std::cout << "Client connected..." << std::endl;

        std::thread(handleClient, clientSocket).detach();
    }

    closesocket(listenSocket);
    WSACleanup();
    return 0;
}
#endif


#if 0
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>

#pragma comment(lib, "Ws2_32.lib")

int main() {
    // Initialize Winsock
    WSADATA wsaData;
    int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        std::cerr << "WSAStartup failed: " << iResult << std::endl;
        return 1;
    }

    // Create a socket
    SOCKET listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // Set up the sockaddr_in structure
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;  // Bind to any available interface
    serverAddr.sin_port = htons(8080);  // Port 8080

    // Bind the socket
    iResult = bind(listenSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (iResult == SOCKET_ERROR) {
        std::cerr << "Bind failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    // Listen for incoming connections
    iResult = listen(listenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        std::cerr << "Listen failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    std::cout << "Server is listening on port 8080..." << std::endl;

    // Accept client connections
    SOCKET clientSocket;
    sockaddr_in clientAddr;
    int clientAddrSize = sizeof(clientAddr);

    while (true) {
        clientSocket = accept(listenSocket, (SOCKADDR*)&clientAddr, &clientAddrSize);
        if (clientSocket == INVALID_SOCKET) {
            std::cerr << "Accept failed: " << WSAGetLastError() << std::endl;
            closesocket(listenSocket);
            WSACleanup();
            return 1;
        }

        // Convert the client IP to a readable format and display it
        char clientIp[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, clientIp, INET_ADDRSTRLEN);
        std::cout << "Accepted connection from: " << clientIp << std::endl;

        // Send a welcome message to the client
        const char* welcomeMessage = "Welcome to the server!\n";
        send(clientSocket, welcomeMessage, strlen(welcomeMessage), 0);

        // Receive data from the client
        char recvbuf[512];
        int recvbuflen = 512;
        int bytesReceived = recv(clientSocket, recvbuf, recvbuflen, 0);
        if (bytesReceived > 0) {
            std::cout << "Received data: " << std::string(recvbuf, bytesReceived) << std::endl;
            Sleep(10 * 1000);
        }
        else if (bytesReceived == 0) {
            std::cout << "Connection closing..." << std::endl;
        }
        else {
            std::cerr << "recv failed: " << WSAGetLastError() << std::endl;
        }

        // Close the client socket
        iResult = shutdown(clientSocket, SD_SEND);
        if (iResult == SOCKET_ERROR) {
            std::cerr << "Shutdown failed: " << WSAGetLastError() << std::endl;
            closesocket(clientSocket);
            WSACleanup();
            return 1;
        }

        closesocket(clientSocket);
    }

    // Cleanup
    closesocket(listenSocket);
    WSACleanup();
    return 0;
}
#endif