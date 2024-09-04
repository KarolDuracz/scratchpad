#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <stdlib.h>  // For rand()

#pragma comment(lib, "Ws2_32.lib")

#define FRAME_SIZE 4096  // Assume each "frame" is 4 KB of data
#define NUM_FRAMES 100   // Number of frames to send

int main() {
    // Initialize Winsock
    WSADATA wsaData;
    int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        std::cerr << "WSAStartup failed: " << iResult << std::endl;
        return 1;
    }

    // Create a socket
    SOCKET sendSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sendSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // Set up the sockaddr_in structure
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(8080);
    inet_pton(AF_INET, "127.0.0.1", &serverAddr.sin_addr);

    // Connect to server
    iResult = connect(sendSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (iResult == SOCKET_ERROR) {
        std::cerr << "Connect failed: " << WSAGetLastError() << std::endl;
        closesocket(sendSocket);
        WSACleanup();
        return 1;
    }

    std::cout << "Connected to server. Sending video frames..." << std::endl;

    char frame[FRAME_SIZE];

    // Simulate sending video frames
    for (int i = 0; i < NUM_FRAMES; ++i) {
        // Generate random data to simulate a video frame
        for (int j = 0; j < FRAME_SIZE; ++j) {
            frame[j] = rand() % 256;
        }

        // Send the frame to the server
        iResult = send(sendSocket, frame, FRAME_SIZE, 0);
        if (iResult == SOCKET_ERROR) {
            std::cerr << "Send failed: " << WSAGetLastError() << std::endl;
            closesocket(sendSocket);
            WSACleanup();
            return 1;
        }

        std::cout << "Sent frame " << i + 1 << " of size " << FRAME_SIZE << " bytes" << std::endl;
    }

    // Shutdown the connection since we're done
    iResult = shutdown(sendSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        std::cerr << "Shutdown failed: " << WSAGetLastError() << std::endl;
        closesocket(sendSocket);
        WSACleanup();
        return 1;
    }

    // Cleanup
    closesocket(sendSocket);
    WSACleanup();
    return 0;
}
