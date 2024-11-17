#include <ntddk.h> // WDK headers

// Function prototypes
VOID UnloadRoutine(_In_ PDRIVER_OBJECT DriverObject);
VOID ExecutePrivilegedInstructions(void);

// Declare external assembly function
extern void DoCli(void);

// DriverEntry function
NTSTATUS DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath)
{
    UNREFERENCED_PARAMETER(RegistryPath);

    KdPrint(("PrivilegedInstructionsDriver loaded.\n"));

    // Set unload routine
    DriverObject->DriverUnload = UnloadRoutine;

    // Execute privileged instructions
    ExecutePrivilegedInstructions();

    return STATUS_SUCCESS;
}

// Unload routine
VOID UnloadRoutine(_In_ PDRIVER_OBJECT DriverObject)
{
    UNREFERENCED_PARAMETER(DriverObject);
    DbgPrint(("PrivilegedInstructionsDriver unloaded.\n"));
}

// Function to execute privileged instructions
VOID ExecutePrivilegedInstructions(void)
{
    unsigned __int64 msrValue;
	unsigned __int64 cr0Value;

    // Example: Read from MSR 0xC0000080 (EFER - Extended Feature Enable Register)
    msrValue = __readmsr(0xC0000080); // MSR intrinsic
	DbgPrint("READ test");
    //DbgPrint(("EFER MSR Value: 0x%llx\n", msrValue));
	//DbgPrint(msrValue);
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "EFER MSR Value: 0x%llx\n", msrValue);
	DbgPrint("READ test end");
	
	DbgPrint("READ CR0");
	cr0Value = __readcr0(); 
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "CR 0 status Value: 0x%llx\n", cr0Value);

    // Example: Attempt CLI (privileged instruction)
    __try {
        DoCli(); // Call external assembly function
        DbgPrint(("CLI instruction executed successfully.\n"));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        DbgPrint(("CLI instruction caused a privileged instruction exception.\n"));
    }

    // Example: Write to MSR (if required)
    __writemsr(0xC0000080, msrValue); // Write the same value back
    DbgPrint(("EFER MSR updated (write test).\n"));
}
