#include <windows.h>
#include <stdio.h>

#define SHARED_MEMORY_NAME "Local\\MySharedMemory"
#define SHARED_MEMORY_SIZE 512

void read_shared_memory() {
    // Open the shared memory block created by the Writer
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_READ, FALSE, SHARED_MEMORY_NAME);
    if (hMapFile == NULL) {
        printf("Could not open file mapping object (Error %d)\n", GetLastError());
        return;
    }

    // Map a view of the shared memory
    char* pBuf = (char*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, SHARED_MEMORY_SIZE);
    if (pBuf == NULL) {
        printf("Could not map view of file (Error %d)\n", GetLastError());
        CloseHandle(hMapFile);
        return;
    }

    printf("Reader Process: Reading data from shared memory every second...\n");

    // Continuously read the data from shared memory every second
    while (1) {
        printf("Reader: Read from shared memory: ");
        for (int i = 0; i < SHARED_MEMORY_SIZE; i++) {
            printf("|%02X %c", (unsigned char)pBuf[i], (unsigned char)pBuf[i]);  // Print as hex
        }
        printf("\n");
        Sleep(1000);  // Read every 1 second
    }

    // Cleanup (unreachable in this example, as it runs indefinitely)
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
}

int main() {
    read_shared_memory();
    return 0;
}
