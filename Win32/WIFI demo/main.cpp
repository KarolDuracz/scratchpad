#include <ws2tcpip.h>

#include <windows.h>
#include <wlanapi.h>
#include <objbase.h>
#include <wtypes.h>
#include <iostream>
#include <string>
#include <wlantypes.h>
#include <stdio.h>

#include <iphlpapi.h>

#include <netioapi.h>

//#include <winsock2.h>
//#include <ws2tcpip.h>




// Link with Wlanapi.lib
#pragma comment(lib, "wlanapi.lib")
#pragma comment(lib, "ole32.lib")

#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "Ws2_32.lib")

void MonitorNetworkStatistics();

void FetchHttpStream(const std::string& cameraIP) {

    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET) {
        std::cerr << "Error creating socket." << std::endl;
        return;
    }

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(217);  // Replace with your desired port
    addr.sin_addr.s_addr = INADDR_ANY; // Listen on all interfaces

    if (bind(udpSocket, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "Error binding socket." << std::endl;
        closesocket(udpSocket);
        return;
    }

    char buffer[1024];  // Buffer to store received data
    sockaddr_in senderAddr;
    int senderAddrSize = sizeof(senderAddr);

    while (true) {
        int bytesReceived = recvfrom(udpSocket, buffer, sizeof(buffer), 0,
            (struct sockaddr*)&senderAddr, &senderAddrSize);
        if (bytesReceived > 0) {
            std::cout << "Received UDP packet: ";
            for (int i = 0; i < bytesReceived; ++i) {
                std::cout << std::hex << (int)(unsigned char)buffer[i] << " ";
            }
            std::cout << std::dec << std::endl;
        }
    }

    closesocket(udpSocket);
    WSACleanup();



}

#if 0
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
#endif

void GetConnectedNetworkInfo() {
    // Step 1: Initialize WLAN handle
    HANDLE hClient = NULL;
    DWORD dwMaxClient = 2; // Version number for the API
    DWORD dwCurVersion = 0;
    DWORD dwResult = 0;

    // Open the WLAN client handle
    dwResult = WlanOpenHandle(dwMaxClient, NULL, &dwCurVersion, &hClient);
    if (dwResult != ERROR_SUCCESS) {
        std::cerr << "WlanOpenHandle failed with error: " << dwResult << std::endl;
        return;
    }

    // Step 2: Enumerate WLAN interfaces
    PWLAN_INTERFACE_INFO_LIST pIfList = NULL;
    dwResult = WlanEnumInterfaces(hClient, NULL, &pIfList);
    if (dwResult != ERROR_SUCCESS) {
        std::cerr << "WlanEnumInterfaces failed with error: " << dwResult << std::endl;
        WlanCloseHandle(hClient, NULL);
        return;
    }

    // Step 3: Retrieve the first connected interface (assuming only one interface is active)
    if (pIfList->dwNumberOfItems == 0) {
        std::cerr << "No WLAN interfaces found." << std::endl;
        WlanFreeMemory(pIfList);
        WlanCloseHandle(hClient, NULL);
        return;
    }

    // Assume the first interface is the one we are interested in
    PWLAN_INTERFACE_INFO pInterfaceInfo = &pIfList->InterfaceInfo[0];
    std::wcout << L"Connected to SSID: " << pInterfaceInfo->InterfaceGuid.Data1 << std::endl;

    // Step 4: Query current connection
    WLAN_CONNECTION_ATTRIBUTES *connAttr = (WLAN_CONNECTION_ATTRIBUTES*)malloc(sizeof(WLAN_CONNECTION_ATTRIBUTES));
    DWORD connAttrSize = sizeof(WLAN_CONNECTION_ATTRIBUTES);
    dwResult = WlanQueryInterface(hClient, &pInterfaceInfo->InterfaceGuid, wlan_intf_opcode_current_connection, NULL, &connAttrSize, (PVOID*)connAttr, NULL);
    if (dwResult != ERROR_SUCCESS) {
        std::cerr << "WlanQueryInterface failed with error: " << dwResult << std::endl;
        WlanFreeMemory(pIfList);
        WlanCloseHandle(hClient, NULL);
        return;
    }

    // Print SSID of the current connection
    std::wcout << L"SSID of the current connection: " << connAttr->strProfileName << std::endl;

    // Step 5: Get the IP address of the current connected device using GetAdaptersInfo or GetAdapterAddresses

    // Using GetAdaptersInfo to retrieve the local IP address
    IP_ADAPTER_INFO adapterInfo[16];
    DWORD dwBufLen = sizeof(adapterInfo);
    DWORD dwRetVal = GetAdaptersInfo(adapterInfo, &dwBufLen);
    if (dwRetVal != ERROR_SUCCESS) {
        std::cerr << "GetAdaptersInfo failed with error: " << dwRetVal << std::endl;
        WlanFreeMemory(pIfList);
        WlanCloseHandle(hClient, NULL);
        return;
    }

    //// Find the adapter corresponding to the connected network
    //for (PIP_ADAPTER_INFO pAdapter = adapterInfo; pAdapter != NULL; pAdapter = pAdapter->Next) {
    //    if (pAdapter->Index == pInterfaceInfo->InterfaceIndex) {
    //        std::cout << "IP Address: " << pAdapter->IpAddressList.IpAddress.String << std::endl;
    //        break;
    //    }
    //}

    // Clean up
    WlanFreeMemory(pIfList);
    WlanCloseHandle(hClient, NULL);
}


void PrintNetworkInfo(const WLAN_AVAILABLE_NETWORK& network) {
    std::cout << L"SSID: ";
    if (network.dot11Ssid.uSSIDLength > 0) {
        std::cout << (network.dot11Ssid.ucSSID, network.dot11Ssid.uSSIDLength);
    }
    else {
        std::cout << L"<Hidden>";
    }
    std::wcout << std::endl;

    std::cout << L"Signal Quality: " << network.wlanSignalQuality << L"%" << std::endl;

    std::cout << L"Authentication: ";
    switch (network.dot11DefaultAuthAlgorithm) {
    case DOT11_AUTH_ALGO_80211_OPEN: std::cout << L"Open"; break;
    case DOT11_AUTH_ALGO_80211_SHARED_KEY: std::cout << L"Shared"; break;
    case DOT11_AUTH_ALGO_WPA: std::cout << L"WPA"; break;
    case DOT11_AUTH_ALGO_WPA_PSK: std::cout << L"WPA-PSK"; break;
    case DOT11_AUTH_ALGO_WPA3: std::cout << L"WPA2"; break;
    case DOT11_AUTH_ALGO_WPA3_SAE: std::cout << L"WPA2-PSK"; break;
    default: std::cout << L"Other"; break;
    }
    std::cout << std::endl << std::endl;
}


void ListAvailableNetworks() {
    HANDLE wlanHandle = NULL;
    DWORD version = 0;
    DWORD result = WlanOpenHandle(2, NULL, &version, &wlanHandle);
    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to open WLAN handle. Error: " << result << std::endl;
        return;
    }

    PWLAN_INTERFACE_INFO_LIST interfaceList = NULL;
    result = WlanEnumInterfaces(wlanHandle, NULL, &interfaceList);
    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to enumerate WLAN interfaces. Error: " << result << std::endl;
        WlanCloseHandle(wlanHandle, NULL);
        return;
    }

    for (DWORD i = 0; i < interfaceList->dwNumberOfItems; i++) {
        PWLAN_INTERFACE_INFO interfaceInfo = &interfaceList->InterfaceInfo[i];
        std::wcout << L"Interface: " << interfaceInfo->strInterfaceDescription << std::endl;

        PWLAN_AVAILABLE_NETWORK_LIST networkList = NULL;
        result = WlanGetAvailableNetworkList(wlanHandle, &interfaceInfo->InterfaceGuid, 0, NULL, &networkList);
        if (result != ERROR_SUCCESS) {
            std::cerr << "Failed to get network list. Error: " << result << std::endl;
            continue;
        }

        for (DWORD j = 0; j < networkList->dwNumberOfItems; j++) {
            std::cout << j  << std::endl;
            WLAN_AVAILABLE_NETWORK netInfo = networkList->Network[j];
            std::cout << netInfo.dot11Ssid.ucSSID << std::endl;
            //PrintNetworkInfo(networkList->Network[j]);
        }

        if (networkList) {
            WlanFreeMemory(networkList);
        }
    }

    if (interfaceList) {
        WlanFreeMemory(interfaceList);
    }
    WlanCloseHandle(wlanHandle, NULL);
}

#if 0
void ConnectToNetwork(const std::wstring& ssid, const std::wstring& password) {
    HANDLE wlanHandle = NULL;
    DWORD version = 0;
    DWORD result = WlanOpenHandle(2, NULL, &version, &wlanHandle);
    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to open WLAN handle. Error: " << result << std::endl;
        return;
    }

    PWLAN_INTERFACE_INFO_LIST interfaceList = NULL;
    result = WlanEnumInterfaces(wlanHandle, NULL, &interfaceList);
    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to enumerate WLAN interfaces. Error: " << result << std::endl;
        WlanCloseHandle(wlanHandle, NULL);
        return;
    }

    // Use the first network interface for simplicity
    PWLAN_INTERFACE_INFO interfaceInfo = &interfaceList->InterfaceInfo[0];

    // Set up the SSID structure
    DOT11_SSID dot11Ssid = {};
    dot11Ssid.uSSIDLength = static_cast<ULONG>(ssid.size());
    memcpy(dot11Ssid.ucSSID, ssid.c_str(), ssid.size());

    std::wcout << " SSID " << dot11Ssid.ucSSID << std::endl;
    // printf for ULONG (unsigned long) --> %lu
    printf("%s %ws %s %ws %d %lu\n", dot11Ssid.ucSSID, dot11Ssid.ucSSID, ssid, ssid, ssid.size(), ssid.size());

    // Prepare the connection parameters
    WLAN_CONNECTION_PARAMETERS connectionParams = {};
    connectionParams.wlanConnectionMode = wlan_connection_mode_temporary_profile;
    connectionParams.dot11BssType = dot11_BSS_type_infrastructure;
    connectionParams.pDot11Ssid = &dot11Ssid;

    // Create a temporary profile for WPA2-PSK (Pre-Shared Key)
    std::wstring profileXml = L"<?xml version=\"1.0\"?>"
        L"<WLANProfile xmlns=\"http://www.microsoft.com/networking/WLAN/profile/v1\">"
        L"<name>" + ssid + L"</name>"
        L"<SSIDConfig><SSID><name>" + ssid + L"</name></SSID></SSIDConfig>"
        L"<connectionType>ESS</connectionType>"
        L"<connectionMode>manual</connectionMode>"
        L"<MSM><security>"
        L"<authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption></authEncryption>"
        L"<sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>" + password + L"</keyMaterial>"
        L"</sharedKey></security></MSM></WLANProfile>";

    // Set the temporary profile and attempt to connect
    //result = WlanSetProfile(wlanHandle, &interfaceInfo->InterfaceGuid, 0, profileXml.c_str(), NULL, TRUE, NULL, NULL);
    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to set profile. Error: " << result << std::endl;
        //WlanCloseHandle(wlanHandle, NULL);
       // return;
    }

    // Attempt to connect to the network
    result = WlanConnect(wlanHandle, &interfaceInfo->InterfaceGuid, &connectionParams, NULL);
    if (result == ERROR_SUCCESS) {
        std::cout << "Successfully connected to the network: " << std::string(ssid.begin(), ssid.end()) << std::endl;
    }
    else {
        std::cerr << "Failed to connect to the network. Error: " << result << std::endl;
    }

    // Cleanup
    if (interfaceList) {
        WlanFreeMemory(interfaceList);
    }
    WlanCloseHandle(wlanHandle, NULL);
}
#endif

void ConnectToNetwork() {
    const std::wstring ssid = L"X10000_01cf";
    const std::wstring password = L"1234567890";

    HANDLE wlanHandle = NULL;
    DWORD version = 0;
    DWORD result = WlanOpenHandle(2, NULL, &version, &wlanHandle);
    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to open WLAN handle. Error: " << result << std::endl;
        return;
    }

    PWLAN_INTERFACE_INFO_LIST interfaceList = NULL;
    result = WlanEnumInterfaces(wlanHandle, NULL, &interfaceList);
    if (result != ERROR_SUCCESS) {
        std::cerr << "Failed to enumerate WLAN interfaces. Error: " << result << std::endl;
        WlanCloseHandle(wlanHandle, NULL);
        return;
    }

    // Select the first interface (assuming only one for simplicity)
    PWLAN_INTERFACE_INFO interfaceInfo = &interfaceList->InterfaceInfo[0];

    //// Create the profile XML for WPA2-PSK
    //std::wstring profileXml = L"<?xml version=\"1.0\"?>"
    //    L"<WLANProfile xmlns=\"http://www.microsoft.com/networking/WLAN/profile/v1\">"
    //    L"<name>" + ssid + L"</name>"
    //    L"<SSIDConfig><SSID><name>" + ssid + L"</name></SSID></SSIDConfig>"
    //    L"<connectionType>ESS</connectionType>"
    //    L"<connectionMode>manual</connectionMode>"
    //    L"<MSM><security>"
    //    L"<authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption></authEncryption>"
    //    L"<sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>" + password + L"</keyMaterial>"
    //    L"</sharedKey></security></MSM></WLANProfile>";

    //// Set the profile for the selected network
    //result = WlanSetProfile(wlanHandle, &interfaceInfo->InterfaceGuid, 0, profileXml.c_str(), NULL, TRUE, NULL, NULL);
    //if (result != ERROR_SUCCESS) {
    //    std::cerr << "Failed to set profile. Error: " << result << std::endl;
    //    WlanCloseHandle(wlanHandle, NULL);
    //    return;
    //}

    // Prepare connection parameters
    WLAN_CONNECTION_PARAMETERS connectionParams = {};
    connectionParams.wlanConnectionMode = wlan_connection_mode_profile;
    connectionParams.dot11BssType = dot11_BSS_type_infrastructure;
    connectionParams.strProfile = ssid.c_str(); // Use the profile name (SSID)

    std::cout << "test " << std::endl;
    std::cout << ssid.c_str() << std::endl;
    std::wcout << ssid.c_str() << std::endl;

    // Attempt to connect to the network
    result = WlanConnect(wlanHandle, &interfaceInfo->InterfaceGuid, &connectionParams, NULL);
    if (result == ERROR_SUCCESS) {
        std::wcout << L"Successfully connected to the network: " << ssid << std::endl;
        //GetConnectedNetworkInfo();
        //MonitorNetworkStatistics();
        FetchHttpStream("192.168.100.140");
    }
    else {
        std::cerr << "Failed to connect to the network. Error: " << result << std::endl;
    }

    // Cleanup
    if (interfaceList) {
        WlanFreeMemory(interfaceList);
    }
    WlanCloseHandle(wlanHandle, NULL);
}

void MonitorNetworkStatistics() {
    MIB_IFTABLE* ifTable = nullptr;
    DWORD size = 0;

    // Allocate memory for the interface table
    DWORD result = GetIfTable(nullptr, &size, false);
    if (result == ERROR_INSUFFICIENT_BUFFER) {
        ifTable = (MIB_IFTABLE*)malloc(size);
        result = GetIfTable(ifTable, &size, false);
    }

    if (result != NO_ERROR) {
        std::cerr << "Failed to retrieve network interface table. Error: " << result << std::endl;
        free(ifTable);
        return;
    }

    std::cout << "Monitoring network interfaces..." << std::endl;

    while (true) {
        Sleep(1000); // Poll every second

        for (DWORD i = 0; i < ifTable->dwNumEntries; i++) {
            MIB_IFROW& row = ifTable->table[i];

            // Print interface name and traffic stats
            std::wcout << L"Interface: " << row.wszName << std::endl;
            std::cout << "Bytes Sent: " << row.dwOutOctets
                << ", Bytes Received: " << row.dwInOctets << std::endl;
        }
    }

    free(ifTable);
}

int main()
{
    std::wcout << L"Listing available networks..." << std::endl;
    ListAvailableNetworks();

    

   /* std::wcout << L"Enter the SSID of the network you want to connect to: ";
    std::wstring ssid;
    std::getline(std::wcin, ssid);

    std::wcout << L"Enter the password for the network: ";
    std::wstring password;
    std::getline(std::wcin, password);

    ConnectToNetwork(ssid, password);*/

    ConnectToNetwork();

	return 0;
}
