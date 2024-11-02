#include <windows.h>
#include <stdio.h>

#define SHARED_MEMORY_NAME "Local\\MySharedMemory"
#define SHARED_MEMORY_SIZE sizeof(RegisterState)

typedef struct {
    DWORD eax;
    DWORD ebx;
    DWORD ecx;
    DWORD edx;
    DWORD esi;
    DWORD edi;
    DWORD ebp;
    DWORD esp;
    DWORD eip;
} RegisterState;

void read_shared_memory() {
    // Open the shared memory block created by the Writer
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_READ, FALSE, SHARED_MEMORY_NAME);
    if (hMapFile == NULL) {
        printf("Could not open file mapping object (Error %d)\n", GetLastError());
        return;
    }

    // Map a view of the shared memory
    RegisterState* pBuf = (RegisterState*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, SHARED_MEMORY_SIZE);
    if (pBuf == NULL) {
        printf("Could not map view of file (Error %d)\n", GetLastError());
        CloseHandle(hMapFile);
        return;
    }

    printf("Reader Process: Reading register states from shared memory every second...\n");

    // Continuously read the data from shared memory every second
    while (1) {
        printf("Reader: Read register states from shared memory:\n");
        printf("EAX: %08X %d\n", pBuf->eax, pBuf->eax);
        printf("EBX: %08X\n", pBuf->ebx);
        printf("ECX: %08X\n", pBuf->ecx);
        printf("EDX: %08X\n", pBuf->edx);
        printf("ESI: %08X\n", pBuf->esi);
        printf("EDI: %08X\n", pBuf->edi);
        printf("EBP: %08X\n", pBuf->ebp);
        printf("ESP: %08X\n", pBuf->esp);
        printf("EIP: %08X %d\n", pBuf->eip, pBuf->eip);
        printf("\n");

        Sleep(100);  // Read every 1 second
    }

    // Cleanup (unreachable in this example, as it runs indefinitely)
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
}

int main() {
    read_shared_memory();
    return 0;
}
 
