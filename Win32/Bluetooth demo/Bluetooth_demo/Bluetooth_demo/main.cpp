#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <winsock2.h>   // Include winsock2.h first
#include <ws2bth.h>     // Bluetooth-specific extensions
#include <windows.h>    // Standard Windows API
#include <bluetoothapis.h>  // Bluetooth API

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Bthprops.lib")

HANDLE find_bluetooth_radio() {
    HBLUETOOTH_RADIO_FIND hFind = NULL;
    HANDLE hRadio = NULL;
    BLUETOOTH_FIND_RADIO_PARAMS btfrp = { sizeof(BLUETOOTH_FIND_RADIO_PARAMS) };

    hFind = BluetoothFindFirstRadio(&btfrp, &hRadio);
    if (hFind == NULL) {
        printf("BluetoothFindFirstRadio failed with error: %ld\n", GetLastError());
        return NULL;
    }

    printf("find first : %d \n", hFind);

    BLUETOOTH_RADIO_INFO radioInfo = { sizeof(BLUETOOTH_RADIO_INFO) };
    if (BluetoothGetRadioInfo(hRadio, &radioInfo) == ERROR_SUCCESS) {
        wprintf(L"Radio Manufacturer: %d\n", radioInfo.manufacturer);
    }
    else {
        printf("Failed to get Bluetooth radio info.\n");
    }

    printf("%ws \n", radioInfo.szName);

    BluetoothFindRadioClose(hFind);
    return hRadio;
}

int make_device_discoverable(HANDLE hRadio) {
    printf("%d %x \n", hRadio, hRadio);
    int result = 0;

    /*
    int result = BluetoothEnableDiscovery(hRadio, TRUE);
    if (result != ERROR_SUCCESS) {
        printf("BluetoothEnableDiscovery failed with error: %ld\n", GetLastError());
        return 1;
    }
    */

    result = BluetoothEnableIncomingConnections(hRadio, TRUE);
    if (result != ERROR_SUCCESS) {
        printf("BluetoothEnableIncomingConnections failed with error: %ld\n", GetLastError());
        return 1;
    }

    printf("Device is now discoverable.\n");
    return 0;
}

void discover_devices(HANDLE hRadio) {
    BLUETOOTH_DEVICE_SEARCH_PARAMS searchParams = {
        sizeof(BLUETOOTH_DEVICE_SEARCH_PARAMS),
        1,  // Return authenticated devices
        0,  // Return remembered devices
        1,  // Return unknown devices
        1,  // Return connected devices
        1,  // Issue inquiries
        10, // Timeout multiplier
        hRadio
    };

    BLUETOOTH_DEVICE_INFO deviceInfo = { sizeof(BLUETOOTH_DEVICE_INFO) };
    HBLUETOOTH_DEVICE_FIND hFindDevice = BluetoothFindFirstDevice(&searchParams, &deviceInfo);

    if (hFindDevice == NULL) {
        printf("No devices found or BluetoothFindFirstDevice failed with error: %ld\n", GetLastError());
        return;
    }

    do {
        wprintf(L"Found device: %s %d \n", deviceInfo.szName, deviceInfo.Address);
    } while (BluetoothFindNextDevice(hFindDevice, &deviceInfo));

    BluetoothFindDeviceClose(hFindDevice);
}


#if 0
int pair_device(HANDLE hRadio, BLUETOOTH_DEVICE_INFO* deviceInfo) {
    int result = BluetoothAuthenticateDeviceEx(NULL, hRadio, deviceInfo, NULL, MITMProtectionNotRequired);
    if (result != ERROR_SUCCESS) {
        printf("BluetoothAuthenticateDevice failed with error: %ld\n", GetLastError());
        return 1;
    }

    printf("Successfully paired with device: %ls\n", deviceInfo->szName);
    return 0;
}
#endif

int pair_device(HANDLE hRadio, BLUETOOTH_DEVICE_INFO* deviceInfo) {
    // Initialize authentication callback parameters
    BLUETOOTH_AUTHENTICATION_CALLBACK_PARAMS callbackParams = { 0 };
    callbackParams.deviceInfo = *deviceInfo;

    // Attempt to authenticate the device
    int result = BluetoothAuthenticateDeviceEx(
        NULL,            // HWND to owner window
        hRadio,          // Bluetooth radio handle
        deviceInfo,      // Device information
        NULL,            // Callback parameters (NULL for default behavior)
        MITMProtectionNotRequired // MITM protection level
    );

    if (result != ERROR_SUCCESS) {
        printf("BluetoothAuthenticateDeviceEx failed with error: %ld\n", GetLastError());
        return 1;
    }

    printf("Successfully paired with device: %ls\n", deviceInfo->szName);
    return 0;
}

// Function to check if a specific device is paired
BOOL is_device_paired(const BLUETOOTH_ADDRESS* deviceAddress) {
    BLUETOOTH_DEVICE_SEARCH_PARAMS searchParams = {
        sizeof(BLUETOOTH_DEVICE_SEARCH_PARAMS),
        0,  // Return authenticated devices
        1,  // Return remembered devices
        0,  // Return unknown devices
        0,  // Return connected devices
        0,  // Issue inquiries
        10, // Timeout multiplier
        NULL
    };

    BLUETOOTH_DEVICE_INFO deviceInfo = { sizeof(BLUETOOTH_DEVICE_INFO) };
    HBLUETOOTH_DEVICE_FIND hFindDevice = BluetoothFindFirstDevice(&searchParams, &deviceInfo);

    if (hFindDevice == NULL) {
        printf("No paired devices found or BluetoothFindFirstDevice failed with error: %ld\n", GetLastError());
        return FALSE;
    }

    BOOL deviceFound = FALSE;

    do {
        if (deviceInfo.Address.ullLong == deviceAddress->ullLong) {
            printf("Device with address %llx is paired.\n", deviceAddress->ullLong);
            deviceFound = TRUE;
            break;
        }
    } while (BluetoothFindNextDevice(hFindDevice, &deviceInfo));

    BluetoothFindDeviceClose(hFindDevice);
    return deviceFound;
}

// Function to connect to a Bluetooth device
void connect_to_device(const BLUETOOTH_DEVICE_INFO* deviceInfo) {
    // For demonstration, we're using BluetoothAuthenticateDeviceEx here
    // This can be replaced with specific connection logic depending on your needs

    int result = BluetoothAuthenticateDeviceEx(
        NULL,            // HWND to owner window
        NULL,            // Bluetooth radio handle (can be NULL if not used)
        (BLUETOOTH_DEVICE_INFO*)deviceInfo, // Device information
        NULL,            // Callback parameters (NULL for default behavior)
        MITMProtectionNotRequired // MITM protection level
    );

    if (result != ERROR_SUCCESS) {
        printf("BluetoothAuthenticateDeviceEx failed with error: %ld\n", GetLastError());
    }
    else {
        printf("Successfully connected to device: %ls\n", deviceInfo->szName);
    }
}


void discover_devices_and_connect(HANDLE hRadio) {
    BLUETOOTH_DEVICE_SEARCH_PARAMS searchParams = {
        sizeof(BLUETOOTH_DEVICE_SEARCH_PARAMS),
        1,  // Return authenticated devices
        0,  // Return remembered devices
        1,  // Return unknown devices
        1,  // Return connected devices
        1,  // Issue inquiries
        10, // Timeout multiplier
        hRadio
    };

    BLUETOOTH_DEVICE_INFO deviceInfo = { sizeof(BLUETOOTH_DEVICE_INFO) };
    HBLUETOOTH_DEVICE_FIND hFindDevice = BluetoothFindFirstDevice(&searchParams, &deviceInfo);

    if (hFindDevice == NULL) {
        printf("No devices found or BluetoothFindFirstDevice failed with error: %ld\n", GetLastError());
        return;
    }

    int deviceIndex = 0;
    BLUETOOTH_DEVICE_INFO secondDeviceInfo = { sizeof(BLUETOOTH_DEVICE_INFO) };
    BOOL foundSecondDevice = FALSE;

    do {
        wprintf(L"Found device %d: %s\n", deviceIndex + 1, deviceInfo.szName);
        wprintf(L"addres : %llu %x \n", deviceInfo.Address, deviceInfo.Address);

        if (deviceIndex == 1) {  // We want the second device
            secondDeviceInfo = deviceInfo;
            foundSecondDevice = TRUE;
        }
        deviceIndex++;
    } while (BluetoothFindNextDevice(hFindDevice, &deviceInfo));

    BluetoothFindDeviceClose(hFindDevice);

    if (foundSecondDevice) {
        printf("Attempting to connect to the second device...\n");
        // Call a function to connect to the second device
        connect_to_device(&secondDeviceInfo);
    }
    else {
        printf("Second device not found.\n");
    }
}



int main() {
    HANDLE hRadio = find_bluetooth_radio();
    if (hRadio == NULL) {
        printf("x");
        return 1;
    }

    discover_devices_and_connect(hRadio);

    //discover_devices(hRadio);

    //if (make_device_discoverable(hRadio) != 0) {
    //    printf("x1");
    //    return 1;
    //}

    // Assume we want to pair with the first found device
    BLUETOOTH_DEVICE_INFO deviceInfo = { sizeof(BLUETOOTH_DEVICE_INFO) };

    // Dummy address; replace with actual device address
    deviceInfo.Address.ullLong = 0x001122334455;
    wcscpy(deviceInfo.szName, L"Device Name");


    if (is_device_paired(&deviceInfo.Address)) {
        printf("Device is paired.\n");
    }
    else {
        printf("Device is not paired.\n");
    }



    if (make_device_discoverable(hRadio) != 0) {
       printf("x1");
        return 1;
    }



    if (pair_device(hRadio, &deviceInfo) != 0) {
        return 1;
    }

    return 0;
}

