#include <ntddk.h>

// Forward declaration of UnloadDriver
VOID UnloadDriver(PDRIVER_OBJECT DriverObject);

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
    UNREFERENCED_PARAMETER(RegistryPath);
    //KdPrint(("Hello, World from Kernel Driver!\n"));
    //DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL, "Hello, World from Kernel Driver!\n");
	DbgPrint("Hello world");
	
    // Set the Unload function
    //DriverObject->DriverUnload = UnloadDriver;

    return STATUS_SUCCESS;
}

VOID UnloadDriver(PDRIVER_OBJECT DriverObject) {
    UNREFERENCED_PARAMETER(DriverObject);
   // KdPrint(("Driver Unloading...\n"));
   //DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL, "Driver Unloading...\n");
}
