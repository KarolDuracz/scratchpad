#include <ntifs.h>
#include <Wdm.h>
#include <ntddk.h> // WDK headers

// Function prototypes
VOID UnloadRoutine(_In_ PDRIVER_OBJECT DriverObject);
VOID ExecutePrivilegedInstructions(void);

// Declare external assembly function
extern void DoCli(void);
extern void StartVMX(void);
extern void checkVMX(void);

// extern 
extern unsigned __int64 _vartest;
extern unsigned __int64 _testVMX;


// Function to allocate memory in a target process
VOID AllocateMemoryInProcess(PEPROCESS TargetProcess) {
    NTSTATUS status;
    PVOID allocatedMemory = NULL;
    SIZE_T size = 4096;  // 4 KB page
    KAPC_STATE apcState;

    // Attach to the target process to manipulate its virtual address space
    KeStackAttachProcess(TargetProcess, &apcState);

    // Allocate virtual memory in the target process
    status = ZwAllocateVirtualMemory(
        ZwCurrentProcess(), &allocatedMemory, 0, &size, 
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
    );

    if (NT_SUCCESS(status)) {
        DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "Memory allocated at: %p\n", allocatedMemory);
    } else {
        DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "ZwAllocateVirtualMemory failed with status: 0x%X\n", status);
    }
	
	LARGE_INTEGER delay;
    delay.QuadPart = -10000 * 10000; // Sleep for 2000ms (2 seconds)
	// Sleep for 2 seconds before memory allocation
    KeDelayExecutionThread(KernelMode, FALSE, &delay);
	
	//unsigned char s = __vmx_on((unsigned __int64)allocatedMemory);
	
	//if (s) {
		//DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "WMX failed with status: %p\n", s);
	//	KdPrint(("WMX failed.\n"));
	//}
	
	StartVMX();
	
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "Test VMX value at: 0x%llx\n", _testVMX);
	
	// read CR4 again after change value from StartVMX()
	unsigned __int64 cr4;
	DbgPrint("READ CR4");
	cr4 = __readcr4(); 
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "CR 4 status Value: 0x%llx\n", cr4);
	
	checkVMX();
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "Test VMX value at: 0x%llx\n", _testVMX);
	
	unsigned char s = __vmx_on(allocatedMemory);
	
	if (s) {
		//DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "WMX failed with status: %p\n", s);
		KdPrint(("WMX failed.\n"));
	}

    // Detach from the process
    KeUnstackDetachProcess(&apcState);
}

#define VGA_MISC_OUTPUT      0x3C2
#define VGA_SEQUENCER_INDEX  0x3C4
#define VGA_SEQUENCER_DATA   0x3C5
#define VGA_GRAPHICS_INDEX   0x3CE
#define VGA_GRAPHICS_DATA    0x3CF
#define VGA_CRTC_INDEX       0x3D4
#define VGA_CRTC_DATA        0x3D5

/* read VGA registers */
// Function to read from VGA register using I/O ports
UINT8 ReadVgaRegister(UINT8 index) {
    __outbyte(VGA_CRTC_INDEX, index);  // Write index to 0x3D4
    return __inbyte(VGA_CRTC_DATA);    // Read data from 0x3D5
}

// Function to read multiple registers
VOID ReadVgaRegisters() {
    UINT8 r00 = ReadVgaRegister(0x00);
    UINT8 r01 = ReadVgaRegister(0x01);
    UINT8 r02 = ReadVgaRegister(0x02);
    UINT8 r03 = ReadVgaRegister(0x03);
    UINT8 r04 = ReadVgaRegister(0x04);
    UINT8 r05 = ReadVgaRegister(0x05);
    UINT8 r06 = ReadVgaRegister(0x06);
    UINT8 r07 = ReadVgaRegister(0x07);
    UINT8 r12 = ReadVgaRegister(0x12);

   DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "VGA CRTC Registers: 00=%02X, 01=%02X, 02=%02X, 03=%02X, 04=%02X, 05=%02X, 06=%02X, 07=%02X, 12=%02X\n",
             r00, r01, r02, r03, r04, r05, r06, r07, r12);
}

// Function to read VGA Miscellaneous Output Register
UINT8 ReadVgaMiscOutput() {
    return __inbyte(0x3CC);  // Read from 0x3CC
}

// Function to write to VGA Miscellaneous Output Register
VOID WriteVgaMiscOutput(UINT8 value) {
    __outbyte(0x3C2, value); // Write to 0x3C2
}

/* PIPE A */
#define INTEL_MMIO_BASE  0xDD400000  // Common MMIO Base for Intel Graphics
#define PIPE_SRCDIM_A    0x6001C     // Active resolution
#define HTOTAL_A         0x60000     // Horizontal timing
//#define VTOTAL_A         0x60004     // Vertical timing 		<---- this is not vertical timing register but HBLANK_A - pipe A horizontal Blank
#define VTOTAL_A         0x6000C	// Vertical total 
#define DPLL_CTRL1       0x06014    // Pixel Clock PLL
#define PIPECONF_A       0x70008     // Pipe Configuration
//#define FBC_STATUS       0x320000    // Framebuffer compression status

/* PIPE B */
#define PIPE_SRCDIM_B    0x6101C     // Active resolution
#define HTOTAL_B         0x61000     // Horizontal timing
#define VTOTAL_B         0x6100C     // Vertical timing
#define DPLL_CTRL2       0x06018    // Pixel Clock PLL
#define PIPECONF_B       0x71008     // Pipe B Configuration

volatile UINT32* MmioBase = NULL;

// Map Intel MMIO Space
NTSTATUS MapIntelMmio() {
    PHYSICAL_ADDRESS MmioPhysAddr;
    MmioPhysAddr.QuadPart = INTEL_MMIO_BASE;
    
    MmioBase = (volatile UINT32*)MmMapIoSpace(MmioPhysAddr, 0x100000, MmNonCached);
    if (!MmioBase) return STATUS_UNSUCCESSFUL;
    return STATUS_SUCCESS;
}

// Read an Intel MMIO register
UINT32 ReadIntelRegister(UINT32 offset) {
    if (!MmioBase) return 0;
    return *(volatile UINT32*)(MmioBase + (offset / 4));
}

// Get Display Resolution (Width x Height)
VOID GetDisplayResolution(UINT32* width, UINT32* height) {
    UINT32 res = ReadIntelRegister(PIPE_SRCDIM_A);
    *width  = (res & 0xFFFF) + 1;        // Bits 0-15 = Width -1
    *height = ((res >> 16) & 0xFFFF) + 1; // Bits 16-31 = Height -1
}

// Function to write to an Intel GPU MMIO register
VOID WriteGfxRegister(UINT32 Offset, UINT32 Value) {
    if (MmioBase) {
        *(volatile UINT32*)(MmioBase + (Offset / 4)) = Value;
        DbgPrint("WriteGfxRegister:\n");
    } else {
        DbgPrint("Error: MMIO space not mapped!\n");
    }
}

// DriverEntry function
NTSTATUS DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath)
{
    UNREFERENCED_PARAMETER(RegistryPath);

    KdPrint(("PrivilegedInstructionsDriver loaded.\n"));

	if (MapIntelMmio() == STATUS_SUCCESS) {
		/* PIPE A */
        UINT32 width, height;
        GetDisplayResolution(&width, &height);
		UINT32 res1 = ReadIntelRegister(HTOTAL_A);
		UINT32 res2 = ReadIntelRegister(VTOTAL_A);
		UINT32 res3 = ReadIntelRegister(DPLL_CTRL1);
		UINT32 res4 = ReadIntelRegister(PIPE_SRCDIM_A);
		UINT32 res5 = ReadIntelRegister(PIPECONF_A);
		DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "---> Value: 0x%llx 0x%llx %x %x %x %x %x\n", width, height, res1, res2, res3, res4, res5);
		
		/* PIPE B */
        GetDisplayResolution(&width, &height);
		UINT32 res1b = ReadIntelRegister(HTOTAL_B);
		UINT32 res2b = ReadIntelRegister(VTOTAL_B);
		UINT32 res3b = ReadIntelRegister(DPLL_CTRL2);
		UINT32 res4b = ReadIntelRegister(PIPE_SRCDIM_B);
		UINT32 res5b = ReadIntelRegister(PIPECONF_B);
		DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "---> Value: 0x%llx 0x%llx %x %x %x %x %x\n", width, height, res1b, res2b, res3b, res4b, res5b);
		
	}
	
	//WriteGfxRegister(PIPE_SRCDIM_A, 0x59f0383); // 1420 x 900 px
	//WriteGfxRegister(PIPE_SRCDIM_A, 0x63f0383); // 1600 x 900 px
	

	// allocate for test some memory on process ID 4 - System. The most important process probably.
	//PEPROCESS SystemProcess;
	//if (NT_SUCCESS(PsLookupProcessByProcessId((HANDLE)4, &SystemProcess))) {
	//	AllocateMemoryInProcess(SystemProcess);
   //     ObDereferenceObject(SystemProcess);
	//}

    // Execute privileged instructions
    //ExecutePrivilegedInstructions();
	
	// read VGA registers
	//ReadVgaRegisters();
	
	// Read current value
    //UINT8 miscVal = ReadVgaMiscOutput();
    //DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "Current VGA Misc Output Register (0x3CC) = 0x%02X\n", miscVal);
	
    // Modify and write a new value (example: enable 25 MHz clock)
    //WriteVgaMiscOutput(miscVal | 0x01);

    // Read again after modification
    //UINT8 newMiscVal = ReadVgaMiscOutput();
    //DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "New VGA Misc Output Register (0x3CC) = 0x%02X\n", newMiscVal);
	
	// Set unload routine
    DriverObject->DriverUnload = UnloadRoutine;
	
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
	unsigned __int64 eflgas_state;
	unsigned __int64 cr4;

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
	
	DbgPrint("READ CR4");
	cr4 = __readcr4(); 
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "CR 4 status Value: 0x%llx\n", cr4);
	
	DbgPrint("eflags");
	eflgas_state = __readeflags();
	DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "eflags status Value: 0x%llx\n", eflgas_state);
	
	// 19-12-2024 Read some power management 
	//msrValue = __readmsr(0xe7); 
	//DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "Value: 0x%llx\n", msrValue);

    // Example: Attempt CLI (privileged instruction)
    __try {
        DoCli(); // Call external assembly function
		
		DbgPrint("eflags");
		eflgas_state = __readeflags();
		DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "eflags status Value: 0x%llx\n", eflgas_state);
		
		DbgPrintEx(DPFLTR_DEFAULT_ID, DPFLTR_INFO_LEVEL, "_vartest: 0x%llx\n", _vartest);
		
        DbgPrint(("CLI instruction executed successfully.\n"));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        DbgPrint(("CLI instruction caused a privileged instruction exception.\n"));
    }

    // Example: Write to MSR (if required)
    __writemsr(0xC0000080, msrValue); // Write the same value back
    DbgPrint(("EFER MSR updated (write test).\n"));
	

	
}
