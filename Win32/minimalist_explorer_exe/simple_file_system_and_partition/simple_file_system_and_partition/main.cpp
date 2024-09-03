/*
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bcrypt.h>


#pragma comment(lib, "zlib.lib")
#pragma comment(lib, "bcrypt.lib")

#define VIRTUAL_DISK_NAME "virtual_disk.bin"
#define PARTITION_COUNT 3
#define PARTITION_SIZE 1024 * 1024 * 10 // 10 MB per partition
#define MAX_FILE_SIZE 1024 * 1024 // 1 MB max file size
#define AES_KEY_SIZE 16 // 128-bit key for AES encryption
#define AES_BLOCK_SIZE 16

typedef struct {
    HANDLE hDisk;
    LARGE_INTEGER partitionOffsets[PARTITION_COUNT];
    BOOL isWriteProtected;
    BYTE encryptionKey[AES_KEY_SIZE];
} VirtualDisk;

// Initialize and generate a random encryption key
void generateEncryptionKey(VirtualDisk* vdisk) {
    if (!BCRYPT_SUCCESS(BCryptGenRandom(NULL, vdisk->encryptionKey, AES_KEY_SIZE, BCRYPT_USE_SYSTEM_PREFERRED_RNG))) {
        printf("Failed to generate encryption key.\n");
        exit(1);
    }
}

// Encrypt data using AES
BOOL encryptData(const BYTE* key, BYTE* data, DWORD dataSize) {
    BCRYPT_ALG_HANDLE hAlgorithm;
    BCRYPT_KEY_HANDLE hKey;
    DWORD result, blockLen, dataLen = dataSize;

    if (!BCRYPT_SUCCESS(BCryptOpenAlgorithmProvider(&hAlgorithm, BCRYPT_AES_ALGORITHM, NULL, 0))) {
        return FALSE;
    }

    if (!BCRYPT_SUCCESS(BCryptSetProperty(hAlgorithm, BCRYPT_CHAINING_MODE, (PUCHAR)BCRYPT_CHAIN_MODE_ECB, sizeof(BCRYPT_CHAIN_MODE_ECB), 0))) {
        BCryptCloseAlgorithmProvider(hAlgorithm, 0);
        return FALSE;
    }

    if (!BCRYPT_SUCCESS(BCryptGenerateSymmetricKey(hAlgorithm, &hKey, NULL, 0, (PUCHAR)key, AES_KEY_SIZE, 0))) {
        BCryptCloseAlgorithmProvider(hAlgorithm, 0);
        return FALSE;
    }

    if (!BCRYPT_SUCCESS(BCryptEncrypt(hKey, data, dataLen, NULL, NULL, 0, data, dataLen, &blockLen, 0))) {
        BCryptDestroyKey(hKey);
        BCryptCloseAlgorithmProvider(hAlgorithm, 0);
        return FALSE;
    }

    BCryptDestroyKey(hKey);
    BCryptCloseAlgorithmProvider(hAlgorithm, 0);
    return TRUE;
}

// Decrypt data using AES
BOOL decryptData(const BYTE* key, BYTE* data, DWORD dataSize) {
    return encryptData(key, data, dataSize); // ECB mode can use same function for encryption and decryption
}

// Protect the virtual disk from write operations
void enableWriteProtection(VirtualDisk* vdisk) {
    vdisk->isWriteProtected = TRUE;
}

// Remove write protection
void disableWriteProtection(VirtualDisk* vdisk) {
    vdisk->isWriteProtected = FALSE;
}

// Compress file data using zlib
int compressData(const char* input, DWORD inputSize, char* output, DWORD* outputSize) {
    uLongf compressedSize = *outputSize;
    if (compress((Bytef*)output, &compressedSize, (const Bytef*)input, inputSize) != Z_OK) {
        return -1;
    }
    *outputSize = compressedSize;
    return 0;
}

// Decompress file data using zlib
int decompressData(const char* input, DWORD inputSize, char* output, DWORD* outputSize) {
    uLongf decompressedSize = *outputSize;
    if (uncompress((Bytef*)output, &decompressedSize, (const Bytef*)input, inputSize) != Z_OK) {
        return -1;
    }
    *outputSize = decompressedSize;
    return 0;
}

void createVirtualDisk(VirtualDisk* vdisk) {
    vdisk->hDisk = CreateFile(
        VIRTUAL_DISK_NAME,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (vdisk->hDisk == INVALID_HANDLE_VALUE) {
        printf("Failed to create virtual disk. Error: %d\n", GetLastError());
        exit(1);
    }

    // Set partition offsets
    for (int i = 0; i < PARTITION_COUNT; i++) {
        vdisk->partitionOffsets[i].QuadPart = i * PARTITION_SIZE;
    }

    // Allocate space for virtual disk
    LARGE_INTEGER diskSize;
    diskSize.QuadPart = PARTITION_COUNT * PARTITION_SIZE;
    SetFilePointerEx(vdisk->hDisk, diskSize, NULL, FILE_BEGIN);
    SetEndOfFile(vdisk->hDisk);

    vdisk->isWriteProtected = FALSE;
    generateEncryptionKey(vdisk);
}

void writeFileToPartition(VirtualDisk* vdisk, int partitionIndex, const char* fileName, const char* data, DWORD dataSize) {
    if (vdisk->isWriteProtected) {
        printf("Disk is write-protected.\n");
        return;
    }

    if (partitionIndex < 0 || partitionIndex >= PARTITION_COUNT) {
        printf("Invalid partition index.\n");
        return;
    }

    // Compress data
    char compressedData[MAX_FILE_SIZE];
    DWORD compressedSize = sizeof(compressedData);
    if (compressData(data, dataSize, compressedData, &compressedSize) != 0) {
        printf("Compression failed.\n");
        return;
    }

    // Encrypt compressed data
    if (!encryptData(vdisk->encryptionKey, (BYTE*)compressedData, compressedSize)) {
        printf("Encryption failed.\n");
        return;
    }

    // Seek to the partition's start
    SetFilePointerEx(vdisk->hDisk, vdisk->partitionOffsets[partitionIndex], NULL, FILE_BEGIN);

    // Write file data
    DWORD bytesWritten;
    WriteFile(vdisk->hDisk, fileName, strlen(fileName), &bytesWritten, NULL);
    WriteFile(vdisk->hDisk, compressedData, compressedSize, &bytesWritten, NULL);

    printf("File '%s' written to partition %d.\n", fileName, partitionIndex);
}

void readFileFromPartition(VirtualDisk* vdisk, int partitionIndex, const char* fileName) {
    if (partitionIndex < 0 || partitionIndex >= PARTITION_COUNT) {
        printf("Invalid partition index.\n");
        return;
    }

    // Seek to the partition's start
    SetFilePointerEx(vdisk->hDisk, vdisk->partitionOffsets[partitionIndex], NULL, FILE_BEGIN);

    // Read file data
    char buffer[MAX_FILE_SIZE];
    DWORD bytesRead;
    ReadFile(vdisk->hDisk, buffer, MAX_FILE_SIZE, &bytesRead, NULL);

    // Decrypt data
    if (!decryptData(vdisk->encryptionKey, (BYTE*)buffer, bytesRead)) {
        printf("Decryption failed.\n");
        return;
    }

    // Decompress data
    char decompressedData[MAX_FILE_SIZE];
    DWORD decompressedSize = sizeof(decompressedData);
    if (decompressData(buffer + strlen(fileName), bytesRead - strlen(fileName), decompressedData, &decompressedSize) != 0) {
        printf("Decompression failed.\n");
        return;
    }

    if (strstr(buffer, fileName) != NULL) {
        printf("File '%s' found in partition %d with content: %s\n", fileName, partitionIndex, decompressedData);
    }
    else {
        printf("File '%s' not found in partition %d.\n", fileName, partitionIndex);
    }
}

void deleteFileFromPartition(VirtualDisk* vdisk, int partitionIndex, const char* fileName) {
    if (vdisk->isWriteProtected) {
        printf("Disk is write-protected.\n");
        return;
    }

    if (partitionIndex < 0 || partitionIndex >= PARTITION_COUNT) {
        printf("Invalid partition index.\n");
        return;
    }

    // Seek to the partition's start
    SetFilePointerEx(vdisk->hDisk, vdisk->partitionOffsets[partitionIndex], NULL, FILE_BEGIN);

    // "Delete" the file by overwriting with zeroes (simple example)
    DWORD bytesWritten;
    char zeroBuffer[MAX_FILE_SIZE] = { 0 };
    WriteFile(vdisk->hDisk, zeroBuffer, MAX_FILE_SIZE, &bytesWritten, NULL);

    printf("File '%s' deleted from partition %d.\n", fileName, partitionIndex);
}

int main() {
    VirtualDisk vdisk;
    createVirtualDisk(&vdisk);

    const char* data = "This is some test data to write into the partition.";
    const char* fileName = "testfile.txt";

    // Write file and measure performance
    writeFileToPartition(&vdisk, 0, fileName, data, strlen(data));

    // Enable write protection
    enableWriteProtection(&vdisk);

    // Attempt to delete file (should fail due to write protection)
    deleteFileFromPartition(&vdisk, 0, fileName);

    // Disable write protection
    disableWriteProtection(&vdisk);

    // Delete file (should succeed now)
    deleteFileFromPartition(&vdisk, 0, fileName);

    // Close the virtual disk
    CloseHandle(vdisk.hDisk);

    return 0;
}
*/




#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <cstdio>
#include <iostream>

#define VIRTUAL_DISK_NAME L"C:\\Windows\\Temp\\virtual_disk.bin"
#define PARTITION_COUNT 3
#define PARTITION_SIZE 1024 * 1024 * 10 // 10 MB per partition
#define MAX_FILE_SIZE 1024 * 1024 // 1 MB max file size

typedef struct {
    HANDLE hDisk;
    LARGE_INTEGER partitionOffsets[PARTITION_COUNT];
} VirtualDisk;

void createVirtualDisk(VirtualDisk* vdisk) {
    vdisk->hDisk = CreateFile(
        VIRTUAL_DISK_NAME,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (vdisk->hDisk == INVALID_HANDLE_VALUE) {
        printf("Failed to create virtual disk. Error: %d\n", GetLastError());
        exit(1);
    }

    // Set partition offsets
    for (int i = 0; i < PARTITION_COUNT; i++) {
        vdisk->partitionOffsets[i].QuadPart = i * PARTITION_SIZE;
    }

    // Allocate space for virtual disk
    LARGE_INTEGER diskSize;
    diskSize.QuadPart = PARTITION_COUNT * PARTITION_SIZE;
    SetFilePointerEx(vdisk->hDisk, diskSize, NULL, FILE_BEGIN);
    SetEndOfFile(vdisk->hDisk);
}

void writeFileToPartition(VirtualDisk* vdisk, int partitionIndex, const char* fileName, const char* data, DWORD dataSize) {
    if (partitionIndex < 0 || partitionIndex >= PARTITION_COUNT) {
        printf("Invalid partition index.\n");
        return;
    }

    // Seek to the partition's start
    SetFilePointerEx(vdisk->hDisk, vdisk->partitionOffsets[partitionIndex], NULL, FILE_BEGIN);

    // Write file data
    DWORD bytesWritten;
    WriteFile(vdisk->hDisk, fileName, strlen(fileName), &bytesWritten, NULL);
    WriteFile(vdisk->hDisk, data, dataSize, &bytesWritten, NULL);
}

void readFileFromPartition(VirtualDisk* vdisk, int partitionIndex, const char* fileName) {
    if (partitionIndex < 0 || partitionIndex >= PARTITION_COUNT) {
        printf("Invalid partition index.\n");
        return;
    }

    // Seek to the partition's start
    SetFilePointerEx(vdisk->hDisk, vdisk->partitionOffsets[partitionIndex], NULL, FILE_BEGIN);

    // Read file data
    char buffer[MAX_FILE_SIZE];
    DWORD bytesRead;
    ReadFile(vdisk->hDisk, buffer, MAX_FILE_SIZE, &bytesRead, NULL);
    buffer[bytesRead] = '\0';

    if (strstr(buffer, fileName) != NULL) {
        printf("File '%s' found in partition %d with content: %s\n", fileName, partitionIndex, buffer + strlen(fileName));
    }
    else {
        printf("File '%s' not found in partition %d.\n", fileName, partitionIndex);
    }
}

void deleteFileFromPartition(VirtualDisk* vdisk, int partitionIndex, const char* fileName) {
    if (partitionIndex < 0 || partitionIndex >= PARTITION_COUNT) {
        printf("Invalid partition index.\n");
        return;
    }

    // Seek to the partition's start
    SetFilePointerEx(vdisk->hDisk, vdisk->partitionOffsets[partitionIndex], NULL, FILE_BEGIN);

    // "Delete" the file by overwriting with zeroes (simple example)
    DWORD bytesWritten;
    char zeroBuffer[MAX_FILE_SIZE] = { 0 };
    WriteFile(vdisk->hDisk, zeroBuffer, MAX_FILE_SIZE, &bytesWritten, NULL);

    printf("File '%s' deleted from partition %d.\n", fileName, partitionIndex);
}

void measurePerformance(const std::function<void()>& operation, const char* operationName) {
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    operation();

    QueryPerformanceCounter(&end);
    double elapsedTime = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    printf("%s took %f ms.\n", operationName, elapsedTime);
}

int main() {
    VirtualDisk vdisk;
    createVirtualDisk(&vdisk);

    const char* data = "This is some test data to write into the partition.";
    const char* fileName = "testfile.txt";

    // Write file and measure performance
    measurePerformance(
        [&] { writeFileToPartition(&vdisk, 0, fileName, data, strlen(data)); },
        "Write file to partition 0"
    );

    std::cout << "test" << std::endl;

    // Read file and measure performance
    measurePerformance(
        [&] { readFileFromPartition(&vdisk, 0, fileName); },
        "Read file from partition 0"
    );

    // Delete file and measure performance
    measurePerformance(
        [&] { deleteFileFromPartition(&vdisk, 0, fileName); },
        "Delete file from partition 0"
    );

    CloseHandle(vdisk.hDisk);

    return 0;
}



/*
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PARTITIONS 4
#define MAX_DISKS 3
#define MAX_FILES 100
#define MAX_FILENAME_LEN 256
#define MAX_FILE_CONTENT_SIZE 1024

typedef struct File {
    char name[MAX_FILENAME_LEN];
    char content[MAX_FILE_CONTENT_SIZE];
} File;

typedef struct Partition {
    char name;
    File* files[MAX_FILES];
    int fileCount;
} Partition;

typedef struct Disk {
    char name;
    Partition* partitions[MAX_PARTITIONS];
    int partitionCount;
} Disk;

// Disks
Disk disks[MAX_DISKS] = {
    {'A', {NULL}, 0},
    {'C', {NULL}, 0},
    {'D', {NULL}, 0}
};

// Create a partition on a disk
void createPartition(Disk* disk, char partName) {
    if (disk->partitionCount >= MAX_PARTITIONS) {
        printf("Disk %c is full, cannot create more partitions.\n", disk->name);
        return;
    }

    Partition* newPartition = (Partition*)malloc(sizeof(Partition));
    newPartition->name = partName;
    newPartition->fileCount = 0;
    disk->partitions[disk->partitionCount++] = newPartition;

    printf("Partition %c: created on disk %c.\n", partName, disk->name);
}

// Find a partition by name on a disk
Partition* findPartition(Disk* disk, char partName) {
    for (int i = 0; i < disk->partitionCount; i++) {
        if (disk->partitions[i]->name == partName) {
            return disk->partitions[i];
        }
    }
    return NULL;
}

// Create a file in a partition
void createFile(Partition* partition, const char* fileName) {
    if (partition->fileCount >= MAX_FILES) {
        printf("Partition %c is full, cannot create more files.\n", partition->name);
        return;
    }

    File* newFile = (File*)malloc(sizeof(File));
    strncpy(newFile->name, fileName, MAX_FILENAME_LEN);
    newFile->content[0] = '\0'; // Empty file
    partition->files[partition->fileCount++] = newFile;

    printf("File '%s' created in partition %c.\n", fileName, partition->name);
}

// Delete a file from a partition
void deleteFile(Partition* partition, const char* fileName) {
    for (int i = 0; i < partition->fileCount; i++) {
        if (strcmp(partition->files[i]->name, fileName) == 0) {
            free(partition->files[i]);
            partition->files[i] = partition->files[--partition->fileCount];
            partition->files[partition->fileCount] = NULL;
            printf("File '%s' deleted from partition %c.\n", fileName, partition->name);
            return;
        }
    }
    printf("File '%s' not found in partition %c.\n", fileName, partition->name);
}

// List files in a partition
void listFiles(Partition* partition) {
    printf("Partition %c:\n", partition->name);
    for (int i = 0; i < partition->fileCount; i++) {
        printf("  [FILE] %s\n", partition->files[i]->name);
    }
}

int main() {
    // Create partitions on disks
    createPartition(&disks[0], '1'); // Partition 1 on Disk A
    createPartition(&disks[1], '1'); // Partition 1 on Disk C
    createPartition(&disks[1], '2'); // Partition 2 on Disk C
    createPartition(&disks[2], '1'); // Partition 1 on Disk D

    // Create files in partitions
    createFile(findPartition(&disks[0], '1'), "fileA1.txt");
    createFile(findPartition(&disks[1], '1'), "fileC1.txt");
    createFile(findPartition(&disks[1], '2'), "fileC2.txt");
    createFile(findPartition(&disks[2], '1'), "fileD1.txt");

    // List files in each partition
    listFiles(findPartition(&disks[0], '1'));
    listFiles(findPartition(&disks[1], '1'));
    listFiles(findPartition(&disks[1], '2'));
    listFiles(findPartition(&disks[2], '1'));

    // Delete a file
    deleteFile(findPartition(&disks[1], '1'), "fileC1.txt");

    // List files again to see the changes
    listFiles(findPartition(&disks[1], '1'));

    return 0;
}
*/