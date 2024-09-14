#define _CRT_SECURE_NO_WARNINGS


#if 1
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define ID_RUN_BUTTON   1
#define ID_STOP_BUTTON  2
#define ID_PAUSE_BUTTON 3
#define ID_STEP_BUTTON  4
#define ID_TEXT_AREA    5
#define ID_OUTPUT_AREA  6
#define ID_REGISTER_AREA 7
#define TIMER_ID        8

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024
#define CLOCK_INTERVAL_MS 500  // Interval for clock signal (500ms -> 2 Hz frequency)

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int instruction_pointer;
    int running;
    int paused;
    int step_mode;
} VM;

// Global VM
VM vm;

// Global output buffer
char output_buffer[2048];
char register_status[256];

// Clock simulation
int clock_cycle = 0;

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 0;
    vm.paused = 0;
    vm.step_mode = 0;
    ZeroMemory(output_buffer, sizeof(output_buffer));
    ZeroMemory(register_status, sizeof(register_status));
}

// Function to append to the output buffer
void append_output(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsprintf(output_buffer + strlen(output_buffer), format, args);
    va_end(args);
}

// Function to append to the register status buffer
void update_register_status() {
    ZeroMemory(register_status, sizeof(register_status));  // Clear the buffer

    for (int i = 0; i < NUM_REGISTERS; i++) {
        char reg_line[32];
        sprintf(reg_line, "R%d: %d\n", i, vm.registers[i]);  // Display each register value
        strcat(register_status, reg_line);
    }
}

// Function to display output in output area
void update_output(HWND hOutputArea) {
    SetWindowTextA(hOutputArea, output_buffer);  // Update the output area with the formatted output
}

// Function to display register status in the register area
void update_register_area(HWND hRegisterArea) {
    update_register_status();
    SetWindowTextA(hRegisterArea, register_status);  // Update the register area with the current status
}

// VM Instruction Set

// MOV Instruction: Move value into register
void instr_mov(int reg, int val) {
    vm.registers[reg] = val;
    append_output("MOV R%d, %d\n", reg, val);
}

// ADD Instruction: Add two registers and store the result in the first register
void instr_add(int reg1, int reg2) {
    vm.registers[reg1] += vm.registers[reg2];
    append_output("ADD R%d, R%d\n", reg1, reg2);
}

// SUB Instruction: Subtract one register from another
void instr_sub(int reg1, int reg2) {
    vm.registers[reg1] -= vm.registers[reg2];
    append_output("SUB R%d, R%d\n", reg1, reg2);
}

// CMP Instruction: Compare two registers (result stored in flags)
void instr_cmp(int reg1, int reg2) {
    if (vm.registers[reg1] == vm.registers[reg2]) {
        append_output("CMP R%d, R%d -> EQUAL\n", reg1, reg2);
    }
    else {
        append_output("CMP R%d, R%d -> NOT EQUAL\n", reg1, reg2);
    }
}

// JMP Instruction: Jump to specific instruction address
void instr_jmp(int addr) {
    append_output("JMP %d -> Jumping to address %d\n", addr, addr);
    vm.instruction_pointer = addr;
}

// Pseudo-compiler: Parse and compile user input code
int compile_vm_code(const char* code) {
    char instruction[256];
    int pc = 0; // Program counter (instruction pointer)

    char* token = strtok((char*)code, "\n");
    int line_number = 1;

    ZeroMemory(output_buffer, sizeof(output_buffer));  // Clear previous output

    while (token != NULL) {
        sscanf(token, "%s", instruction);
        append_output("Line %d: Address %d: ", line_number, pc);  // Show line number and address

        if (strcmp(instruction, "MOV") == 0) {
            int reg, val;
            if (sscanf(token, "MOV R%d, %d", &reg, &val) == 2) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_mov(reg, val);
                }
                else {
                    MessageBox(NULL, L"Invalid register in MOV instruction.", L"Compilation Error", MB_OK | MB_ICONERROR);
                    return 0;
                }
            }
            else {
                MessageBox(NULL, L"Invalid MOV syntax.", L"Compilation Error", MB_OK | MB_ICONERROR);
                return 0;
            }
        }
        else if (strcmp(instruction, "ADD") == 0) {
            int reg1, reg2;
            if (sscanf(token, "ADD R%d, R%d", &reg1, &reg2) == 2) {
                instr_add(reg1, reg2);
            }
            else {
                MessageBox(NULL, L"Invalid ADD syntax.", L"Compilation Error", MB_OK | MB_ICONERROR);
                return 0;
            }
        }
        else if (strcmp(instruction, "SUB") == 0) {
            int reg1, reg2;
            if (sscanf(token, "SUB R%d, R%d", &reg1, &reg2) == 2) {
                instr_sub(reg1, reg2);
            }
            else {
                MessageBox(NULL, L"Invalid SUB syntax.", L"Compilation Error", MB_OK | MB_ICONERROR);
                return 0;
            }
        }
        else if (strcmp(instruction, "CMP") == 0) {
            int reg1, reg2;
            if (sscanf(token, "CMP R%d, R%d", &reg1, &reg2) == 2) {
                instr_cmp(reg1, reg2);
            }
            else {
                MessageBox(NULL, L"Invalid CMP syntax.", L"Compilation Error", MB_OK | MB_ICONERROR);
                return 0;
            }
        }
        else if (strcmp(instruction, "JMP") == 0) {
            int addr;
            if (sscanf(token, "JMP %d", &addr) == 1) {
                instr_jmp(addr);
            }
            else {
                MessageBox(NULL, L"Invalid JMP syntax.", L"Compilation Error", MB_OK | MB_ICONERROR);
                return 0;
            }
        }
        else {
            wchar_t msg[256];
            swprintf(msg, 256, L"Unknown instruction: %S", instruction);
            MessageBox(NULL, msg, L"Compilation Error", MB_OK | MB_ICONERROR);
            return 0;
        }

        token = strtok(NULL, "\n");
        pc += 4;  // Assuming each instruction is 4 bytes in size
        line_number++;
    }

    return 1;  // Compilation successful
}

// VM runner function to execute user program
void execute_vm_step(HWND hOutputArea, HWND hRegisterArea) {
    char instruction[256];
    sprintf(instruction, "Line %d: Address %d: ", vm.instruction_pointer, vm.instruction_pointer * 4);

    // Here, handle the execution of instructions based on the instruction pointer.
    append_output("%s\n", instruction); // Placeholder for actual instruction execution

    // Call output update functions
    update_output(hOutputArea);
    update_register_area(hRegisterArea);  // Update the register status
}

// Function to stop the VM
void stop_vm(HWND hOutputArea, HWND hRegisterArea) {
    vm.running = 0;
    append_output("\nExecution stopped.\n");
    update_output(hOutputArea);
    update_register_area(hRegisterArea);
}

// Function to pause the VM
void pause_vm(HWND hOutputArea, HWND hRegisterArea) {
    vm.paused = 1;
    append_output("\nExecution paused.\n");
    update_output(hOutputArea);
    update_register_area(hRegisterArea);
}

// Function to step through the code
void step_vm(HWND hOutputArea, HWND hRegisterArea) {
    vm.step_mode = 1;
    execute_vm_step(hOutputArea, hRegisterArea);
}

// Window Procedure: Handles messages sent to the window
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hOutputArea, hRegisterArea;

    switch (uMsg) {
    case WM_CREATE:
        // Create input area for user code
        hTextArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL,
            10, 10, 600, 600,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        // Create output area for displaying VM output
        hOutputArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | ES_READONLY | WS_VSCROLL,
            620, 10, 560, 600,
            hwnd, (HMENU)ID_OUTPUT_AREA, GetModuleHandle(NULL), NULL);

        // Create register status area
        hRegisterArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | ES_READONLY,
            620, 620, 560, 120,
            hwnd, (HMENU)ID_REGISTER_AREA, GetModuleHandle(NULL), NULL);

        // Create buttons for control
        CreateWindowEx(0, L"BUTTON", L"Run",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 620, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);

        CreateWindowEx(0, L"BUTTON", L"Stop",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            100, 620, 80, 30,
            hwnd, (HMENU)ID_STOP_BUTTON, GetModuleHandle(NULL), NULL);

        CreateWindowEx(0, L"BUTTON", L"Pause",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            190, 620, 80, 30,
            hwnd, (HMENU)ID_PAUSE_BUTTON, GetModuleHandle(NULL), NULL);

        CreateWindowEx(0, L"BUTTON", L"Step",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            280, 620, 80, 30,
            hwnd, (HMENU)ID_STEP_BUTTON, GetModuleHandle(NULL), NULL);

        SetTimer(hwnd, TIMER_ID, CLOCK_INTERVAL_MS, NULL);  // Set timer for clock ticks
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            reset_vm();
            // Get the code from the input area
            wchar_t code[2048];
            GetWindowText(hTextArea, code, 2048);

            // Convert wide characters to standard char (for simplicity)
            char program_code[2048];
            wcstombs(program_code, code, 2048);

            // Compile and run the user code
            if (compile_vm_code(program_code)) {
                vm.running = 1;
                vm.paused = 0;
                update_output(hOutputArea);
                update_register_area(hRegisterArea);
            }
        }
        else if (LOWORD(wParam) == ID_STOP_BUTTON) {
            stop_vm(hOutputArea, hRegisterArea);
        }
        else if (LOWORD(wParam) == ID_PAUSE_BUTTON) {
            pause_vm(hOutputArea, hRegisterArea);
        }
        else if (LOWORD(wParam) == ID_STEP_BUTTON) {
            step_vm(hOutputArea, hRegisterArea);
        }
        break;

    case WM_TIMER:
        if (wParam == TIMER_ID && vm.running && !vm.paused) {
            // Clock signal tick
            clock_cycle++;
            execute_vm_step(hOutputArea, hRegisterArea);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const wchar_t* CLASS_NAME = L"SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"Simple Virtual Machine",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 800,  // Resize to 1200x800
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif


#if 0
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define ID_RUN_BUTTON   1
#define ID_STOP_BUTTON  2
#define ID_PAUSE_BUTTON 3
#define ID_STEP_BUTTON  4
#define ID_TEXT_AREA    5
#define ID_OUTPUT_AREA  6
#define ID_REGISTER_AREA 7
#define TIMER_ID        8

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024
#define CLOCK_INTERVAL_MS 500  // Interval for clock signal (500ms -> 2 Hz frequency)

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int instruction_pointer;
    int running;
    int paused;
    int step_mode;
} VM;

// Global VM
VM vm;

// Global output buffer
char output_buffer[2048];
char register_status[256];

// Clock simulation
int clock_cycle = 0;

// Line chart buffer
int clock_signal[100]; // Store clock signal history for visualization

// Sample program to test infinite loop
char program_code[] =
"MOV R0, 10\n"
"MOV R1, 20\n"
"JMP 0\n";

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 0;
    vm.paused = 0;
    vm.step_mode = 0;
    ZeroMemory(output_buffer, sizeof(output_buffer));
    ZeroMemory(register_status, sizeof(register_status));
}

// Function to append to the output buffer
void append_output(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsprintf(output_buffer + strlen(output_buffer), format, args);
    va_end(args);
}

// Function to append to the register status buffer
void update_register_status() {
    ZeroMemory(register_status, sizeof(register_status));  // Clear the buffer

    for (int i = 0; i < NUM_REGISTERS; i++) {
        char reg_line[32];
        sprintf(reg_line, "R%d: %d\n", i, vm.registers[i]);  // Display each register value
        strcat(register_status, reg_line);
    }
}

// Function to display output in output area
void update_output(HWND hOutputArea) {
    SetWindowTextA(hOutputArea, output_buffer);  // Update the output area with the formatted output
}

// Function to display register status in the register area
void update_register_area(HWND hRegisterArea) {
    update_register_status();
    SetWindowTextA(hRegisterArea, register_status);  // Update the register area with the current status
}

// VM Instruction Set

// Arithmetic Instructions
void instr_mov(int reg, int val) {
    vm.registers[reg] = val;
    append_output("MOV R%d, %d\n", reg, val);
}

void instr_jmp(int addr) {
    append_output("JMP %d -> Jumping to address %d\n", addr, addr);
    vm.instruction_pointer = addr;
}

// Error message display for pseudo-compiler
void error_message(const wchar_t* message) {
    MessageBox(NULL, message, L"Compilation Error", MB_OK | MB_ICONERROR);
}

// Basic pseudo-compiler function with instruction size and line numbers
int compile_vm_code(const char* code) {
    char instruction[256];
    int pc = 0; // Program counter (instruction pointer)

    char* token = strtok((char*)code, "\n");
    int line_number = 1;

    ZeroMemory(output_buffer, sizeof(output_buffer));  // Clear previous output

    while (token != NULL) {
        sscanf(token, "%s", instruction);
        append_output("Line %d: Address %d: ", line_number, pc);  // Show line number and address

        if (strcmp(instruction, "MOV") == 0) {
            int reg, val;
            if (sscanf(token, "MOV R%d, %d", &reg, &val) == 2) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_mov(reg, val);
                }
                else {
                    error_message(L"Invalid register in MOV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MOV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JMP") == 0) {
            int addr;
            if (sscanf(token, "JMP %d", &addr) == 1) {
                instr_jmp(addr);
            }
            else {
                error_message(L"Invalid JMP syntax.");
                return 0;
            }
        }
        else {
            wchar_t msg[256];
            swprintf(msg, 256, L"Unknown instruction: %S", instruction);
            error_message(msg);
            return 0;
        }

        token = strtok(NULL, "\n");
        pc += 4;  // Assuming each instruction is 4 bytes in size
        line_number++;
    }

    return 1;  // Compilation successful
}

// VM runner function with infinite loop
void execute_vm_step(HWND hOutputArea, HWND hRegisterArea) {
    char instruction[256];
    sprintf(instruction, "Line %d: Address %d: ", vm.instruction_pointer, vm.instruction_pointer * 4);

    switch (vm.instruction_pointer) {
    case 0:
        append_output("%s MOV R0, 10 -->", instruction);
        vm.registers[0] = 10;
        vm.instruction_pointer++;
        break;
    case 1:
        append_output("%s MOV R1, 20 -->", instruction);
        vm.registers[1] = 20;
        vm.instruction_pointer++;
        break;
    case 2:
        append_output("%s JMP 0 -->", instruction);
        vm.instruction_pointer = 0;  // Loop back to address 0
        break;
    default:
        append_output("Unknown instruction at %d\n", vm.instruction_pointer);
        vm.running = 0;
        break;
    }

    append_output("\n");  // Move to the next line for proper formatting
    update_output(hOutputArea);
    update_register_area(hRegisterArea);  // Update the register status
}

// Function to stop the VM
void stop_vm(HWND hOutputArea, HWND hRegisterArea) {
    vm.running = 0;
    append_output("\nExecution stopped.\n");
    update_output(hOutputArea);
    update_register_area(hRegisterArea);
}

// Function to pause the VM
void pause_vm(HWND hOutputArea, HWND hRegisterArea) {
    vm.paused = 1;
    append_output("\nExecution paused.\n");
    update_output(hOutputArea);
    update_register_area(hRegisterArea);
}

// Function to execute one instruction in step mode
void step_vm(HWND hOutputArea, HWND hRegisterArea) {
    if (vm.paused || vm.step_mode) {
        append_output("\nStep execution...\n");
        execute_vm_step(hOutputArea, hRegisterArea);
        update_output(hOutputArea);
    }
}

// Window procedure for handling messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hOutputArea, hRunButton, hStopButton, hPauseButton, hStepButton, hRegisterArea;

    switch (uMsg) {
    case WM_CREATE:
        hTextArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL,
            10, 10, 600, 600,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        hRunButton = CreateWindowEx(0, L"BUTTON", L"Run",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 620, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);

        hStopButton = CreateWindowEx(0, L"BUTTON", L"Stop",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            100, 620, 80, 30,
            hwnd, (HMENU)ID_STOP_BUTTON, GetModuleHandle(NULL), NULL);

        hPauseButton = CreateWindowEx(0, L"BUTTON", L"Pause",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            190, 620, 80, 30,
            hwnd, (HMENU)ID_PAUSE_BUTTON, GetModuleHandle(NULL), NULL);

        hStepButton = CreateWindowEx(0, L"BUTTON", L"Step",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            280, 620, 80, 30,
            hwnd, (HMENU)ID_STEP_BUTTON, GetModuleHandle(NULL), NULL);

        hOutputArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | ES_READONLY | WS_VSCROLL,
            620, 10, 560, 600,
            hwnd, (HMENU)ID_OUTPUT_AREA, GetModuleHandle(NULL), NULL);

        // Register Area - For showing CPU register status
        hRegisterArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | ES_READONLY,
            620, 620, 560, 120,
            hwnd, (HMENU)ID_REGISTER_AREA, GetModuleHandle(NULL), NULL);

        SetTimer(hwnd, TIMER_ID, CLOCK_INTERVAL_MS, NULL);  // Set timer for clock ticks
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            reset_vm();
            compile_vm_code(program_code);  // Load and compile the infinite loop test program
            vm.running = 1;
            vm.paused = 0;
            update_output(hOutputArea);
            update_register_area(hRegisterArea);
        }
        else if (LOWORD(wParam) == ID_STOP_BUTTON) {
            stop_vm(hOutputArea, hRegisterArea);
        }
        else if (LOWORD(wParam) == ID_PAUSE_BUTTON) {
            pause_vm(hOutputArea, hRegisterArea);
        }
        else if (LOWORD(wParam) == ID_STEP_BUTTON) {
            vm.step_mode = 1;
            step_vm(hOutputArea, hRegisterArea);
        }
        break;

    case WM_TIMER:
        if (wParam == TIMER_ID && vm.running && !vm.paused) {
            // Clock signal tick
            clock_cycle++;
            execute_vm_step(hOutputArea, hRegisterArea);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const wchar_t* CLASS_NAME = L"SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"Simple Virtual Machine",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 800,  // Resize to 1200x800
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

#endif




#if 0
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define ID_RUN_BUTTON   1
#define ID_STOP_BUTTON  2
#define ID_PAUSE_BUTTON 3
#define ID_STEP_BUTTON  4
#define ID_TEXT_AREA    5
#define ID_OUTPUT_AREA  6
#define TIMER_ID        7

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024
#define CLOCK_INTERVAL_MS 500  // Interval for clock signal (500ms -> 2 Hz frequency)

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int instruction_pointer;
    int running;
    int paused;
    int step_mode;
} VM;

// Global VM
VM vm;

// Global output buffer
char output_buffer[2048];

// Clock simulation
int clock_cycle = 0;

// Line chart buffer
int clock_signal[100]; // Store clock signal history for visualization

// Sample program to test infinite loop
char program_code[] =
"MOV R0, 10\n"
"MOV R1, 20\n"
"JMP 0\n";

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 0;
    vm.paused = 0;
    vm.step_mode = 0;
    ZeroMemory(output_buffer, sizeof(output_buffer));
}

// Function to append to the output buffer
void append_output(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsprintf(output_buffer + strlen(output_buffer), format, args);
    va_end(args);
}

// Function to display output in output area
void update_output(HWND hOutputArea) {
    SetWindowTextA(hOutputArea, output_buffer);  // Update the output area with the formatted output
}

// VM Instruction Set

// Arithmetic Instructions
void instr_mov(int reg, int val) {
    vm.registers[reg] = val;
    append_output("MOV R%d, %d\n", reg, val);
}

void instr_jmp(int addr) {
    append_output("JMP %d -> Jumping to address %d\n", addr, addr);
    vm.instruction_pointer = addr;
}

// Error message display for pseudo-compiler
void error_message(const wchar_t* message) {
    MessageBox(NULL, message, L"Compilation Error", MB_OK | MB_ICONERROR);
}

// Basic pseudo-compiler function with instruction size and line numbers
int compile_vm_code(const char* code) {
    char instruction[256];
    int pc = 0; // Program counter (instruction pointer)

    char* token = strtok((char*)code, "\n");
    int line_number = 1;

    ZeroMemory(output_buffer, sizeof(output_buffer));  // Clear previous output

    while (token != NULL) {
        sscanf(token, "%s", instruction);
        append_output("Line %d: Address %d: ", line_number, pc);  // Show line number and address

        if (strcmp(instruction, "MOV") == 0) {
            int reg, val;
            if (sscanf(token, "MOV R%d, %d", &reg, &val) == 2) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_mov(reg, val);
                }
                else {
                    error_message(L"Invalid register in MOV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MOV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JMP") == 0) {
            int addr;
            if (sscanf(token, "JMP %d", &addr) == 1) {
                instr_jmp(addr);
            }
            else {
                error_message(L"Invalid JMP syntax.");
                return 0;
            }
        }
        else {
            wchar_t msg[256];
            swprintf(msg, 256, L"Unknown instruction: %S", instruction);
            error_message(msg);
            return 0;
        }

        token = strtok(NULL, "\n");
        pc += 4;  // Assuming each instruction is 4 bytes in size
        line_number++;
    }

    return 1;  // Compilation successful
}

// VM runner function with infinite loop
void execute_vm_step(HWND hOutputArea) {
    char instruction[256];
    sprintf(instruction, "Line %d: Address %d: ", vm.instruction_pointer, vm.instruction_pointer * 4);

    switch (vm.instruction_pointer) {
    case 0:
        append_output("%s MOV R0, 10 -->", instruction);
        vm.registers[0] = 10;
        vm.instruction_pointer++;
        break;
    case 1:
        append_output("%s MOV R1, 20 -->", instruction);
        vm.registers[1] = 20;
        vm.instruction_pointer++;
        break;
    case 2:
        append_output("%s JMP 0 -->", instruction);
        vm.instruction_pointer = 0;  // Loop back to address 0
        break;
    default:
        append_output("Unknown instruction at %d\n", vm.instruction_pointer);
        vm.running = 0;
        break;
    }

    append_output("\n");  // Move to the next line for proper formatting
    update_output(hOutputArea);
}

// Function to stop the VM
void stop_vm(HWND hOutputArea) {
    vm.running = 0;
    append_output("\nExecution stopped.\n");
    update_output(hOutputArea);
}

// Function to pause the VM
void pause_vm(HWND hOutputArea) {
    vm.paused = 1;
    append_output("\nExecution paused.\n");
    update_output(hOutputArea);
}

// Function to execute one instruction in step mode
void step_vm(HWND hOutputArea) {
    if (vm.paused || vm.step_mode) {
        append_output("\nStep execution...\n");
        execute_vm_step(hOutputArea);
        update_output(hOutputArea);
    }
}

// Window procedure for handling messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hOutputArea, hRunButton, hStopButton, hPauseButton, hStepButton;

    switch (uMsg) {
    case WM_CREATE:
        hTextArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL,
            10, 10, 600, 600,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        hRunButton = CreateWindowEx(0, L"BUTTON", L"Run",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 620, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);

        hStopButton = CreateWindowEx(0, L"BUTTON", L"Stop",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            100, 620, 80, 30,
            hwnd, (HMENU)ID_STOP_BUTTON, GetModuleHandle(NULL), NULL);

        hPauseButton = CreateWindowEx(0, L"BUTTON", L"Pause",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            190, 620, 80, 30,
            hwnd, (HMENU)ID_PAUSE_BUTTON, GetModuleHandle(NULL), NULL);

        hStepButton = CreateWindowEx(0, L"BUTTON", L"Step",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            280, 620, 80, 30,
            hwnd, (HMENU)ID_STEP_BUTTON, GetModuleHandle(NULL), NULL);

        hOutputArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL | ES_READONLY,
            620, 10, 560, 600,
            hwnd, (HMENU)ID_OUTPUT_AREA, GetModuleHandle(NULL), NULL);

        // Set timer for clock ticks
        SetTimer(hwnd, TIMER_ID, CLOCK_INTERVAL_MS, NULL);
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            // When the "Run" button is clicked
            reset_vm();
            compile_vm_code(program_code);  // Load and compile the infinite loop test program
            vm.running = 1;
            vm.paused = 0;
            update_output(hOutputArea);
        }
        else if (LOWORD(wParam) == ID_STOP_BUTTON) {
            stop_vm(hOutputArea);
        }
        else if (LOWORD(wParam) == ID_PAUSE_BUTTON) {
            pause_vm(hOutputArea);
        }
        else if (LOWORD(wParam) == ID_STEP_BUTTON) {
            vm.step_mode = 1;
            step_vm(hOutputArea);
        }
        break;

    case WM_TIMER:
        if (wParam == TIMER_ID && vm.running && !vm.paused) {
            // Clock signal tick
            clock_cycle++;
            execute_vm_step(hOutputArea);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const wchar_t* CLASS_NAME = L"SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"Simple Virtual Machine",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 800,  // Resize to 1200x800
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif


#if 0
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <string>
#include <iostream>

#define ID_RUN_BUTTON   1
#define ID_STOP_BUTTON  2
#define ID_PAUSE_BUTTON 3
#define ID_STEP_BUTTON  4
#define ID_TEXT_AREA    5
#define ID_OUTPUT_AREA  6
#define TIMER_ID        7

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024
#define CLOCK_INTERVAL_MS 500  // Interval for clock signal (500ms -> 2 Hz frequency)

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int instruction_pointer;
    int running;
    int paused;
} VM;

// Global VM
VM vm;

// Global output buffer
char output_buffer[2048];

// Clock simulation
int clock_cycle = 0;

// Line chart buffer
int clock_signal[100]; // Store clock signal history for visualization

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 0;
    vm.paused = 0;
    ZeroMemory(output_buffer, sizeof(output_buffer));
}

// Function to append to the output buffer
void append_output(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsprintf(output_buffer + strlen(output_buffer), format, args);
    va_end(args);
}

// Function to display output in output area
void update_output(HWND hOutputArea) {
    SetWindowTextA(hOutputArea, output_buffer);  // Update the output area with the formatted output
}

// VM Instruction Set

// Arithmetic Instructions
void instr_add(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] + vm.registers[src2];
    append_output("ADD R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_sub(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] - vm.registers[src2];
    append_output("SUB R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_mul(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] * vm.registers[src2];
    append_output("MUL R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_div(int dest, int src1, int src2) {
    if (vm.registers[src2] == 0) {
        append_output("Error: Division by zero!\n");
        vm.running = 0;
        return;
    }
    vm.registers[dest] = vm.registers[src1] / vm.registers[src2];
    append_output("DIV R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

// Stack Operations
void instr_push(int reg) {
    if (vm.stack_ptr + 1 >= STACK_SIZE) {
        append_output("Stack Overflow!\n");
        vm.running = 0;
        return;
    }
    vm.stack[++vm.stack_ptr] = vm.registers[reg];
    append_output("PUSH R%d -> Stack[%d] = %d\n", reg, vm.stack_ptr, vm.registers[reg]);
}

void instr_pop(int reg) {
    if (vm.stack_ptr < 0) {
        append_output("Stack Underflow!\n");
        vm.running = 0;
        return;
    }
    vm.registers[reg] = vm.stack[vm.stack_ptr--];
    append_output("POP Stack[%d] -> R%d = %d\n", vm.stack_ptr + 1, reg, vm.registers[reg]);
}

// Branching
void instr_jmp(int addr) {
    vm.instruction_pointer = addr;
    append_output("JMP %d -> Instruction Pointer = %d\n", addr, addr);
}

void instr_jeq(int addr, int reg1, int reg2) {
    if (vm.registers[reg1] == vm.registers[reg2]) {
        vm.instruction_pointer = addr;
        append_output("JEQ -> JMP to %d (R%d == R%d)\n", addr, reg1, reg2);
    }
    else {
        append_output("JEQ -> No jump (R%d != R%d)\n", reg1, reg2);
    }
}

// Comparison Instructions
void instr_cmp(int reg1, int reg2) {
    int result = vm.registers[reg1] - vm.registers[reg2];
    append_output("CMP R%d, R%d -> Result = %d\n", reg1, reg2, result);
}

// Error message display for pseudo-compiler
void error_message(const wchar_t* message) {
    MessageBox(NULL, message, L"Compilation Error", MB_OK | MB_ICONERROR);
}

// Basic pseudo-compiler function with instruction size and line numbers
int compile_vm_code(const char* code) {
    char instruction[256];
    int pc = 0; // Program counter (instruction pointer)

    char* token = strtok((char*)code, "\n");
    int line_number = 1;

    ZeroMemory(output_buffer, sizeof(output_buffer));  // Clear previous output

    while (token != NULL) {
        sscanf(token, "%s", instruction);
        append_output("Line %d: Address %d: ", line_number, pc);  // Show line number and address

        if (strcmp(instruction, "MOV") == 0) {
            int reg, val;
            if (sscanf(token, "MOV R%d, %d", &reg, &val) == 2) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    vm.registers[reg] = val;
                    append_output("MOV R%d, %d\n", reg, val);
                }
                else {
                    error_message(L"Invalid register in MOV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MOV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "ADD") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "ADD R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_add(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in ADD instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid ADD syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "SUB") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "SUB R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_sub(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in SUB instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid SUB syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "MUL") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "MUL R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_mul(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in MUL instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MUL syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "DIV") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "DIV R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_div(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in DIV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid DIV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "CMP") == 0) {
            int reg1, reg2;
            if (sscanf(token, "CMP R%d, R%d", &reg1, &reg2) == 2) {
                if (reg1 >= 0 && reg1 < NUM_REGISTERS && reg2 >= 0 && reg2 < NUM_REGISTERS) {
                    instr_cmp(reg1, reg2);
                }
                else {
                    error_message(L"Invalid register in CMP instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid CMP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "PUSH") == 0) {
            int reg;
            if (sscanf(token, "PUSH R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_push(reg);
                }
                else {
                    error_message(L"Invalid register in PUSH instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid PUSH syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "POP") == 0) {
            int reg;
            if (sscanf(token, "POP R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_pop(reg);
                }
                else {
                    error_message(L"Invalid register in POP instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid POP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JMP") == 0) {
            int addr;
            if (sscanf(token, "JMP %d", &addr) == 1) {
                instr_jmp(addr);
            }
            else {
                error_message(L"Invalid JMP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JEQ") == 0) {
            int addr, reg1, reg2;
            if (sscanf(token, "JEQ %d, R%d, R%d", &addr, &reg1, &reg2) == 3) {
                instr_jeq(addr, reg1, reg2);
            }
            else {
                error_message(L"Invalid JEQ syntax.");
                return 0;
            }
        }
        else {
            wchar_t msg[256];
            swprintf(msg, 256, L"Unknown instruction: %S", instruction);
            error_message(msg);
            return 0;
        }

        token = strtok(NULL, "\n");
        pc += 4;  // Assuming each instruction is 4 bytes in size
        line_number++;
    }

    return 1;  // Compilation successful
}

// VM runner function with infinite loop
void run_vm(const char* code, HWND hOutputArea) {
    reset_vm();
    if (compile_vm_code(code)) {
        // Start VM execution
        vm.running = 1;
        vm.paused = 0;
        update_output(hOutputArea);  // Display the results in the output area
    }
}

// Function to stop the VM
void stop_vm(HWND hOutputArea) {
    vm.running = 0;
    append_output("\nExecution stopped.\n");
    update_output(hOutputArea);
}

// Function to pause the VM
void pause_vm(HWND hOutputArea) {
    vm.paused = 1;
    append_output("\nExecution paused.\n");
    update_output(hOutputArea);
}

// Function to execute one instruction in step mode
void step_vm(HWND hOutputArea) {
    if (vm.paused) {
        append_output("Step instruction execution...\n");
        // Execute the next instruction in step mode (pseudo code needed to execute instruction)
        update_output(hOutputArea);
    }
}

// Window procedure for handling messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hOutputArea, hRunButton, hStopButton, hPauseButton, hStepButton;

    switch (uMsg) {
    case WM_CREATE:
        hTextArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL,
            10, 10, 600, 600,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        hRunButton = CreateWindowEx(0, L"BUTTON", L"Run",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 620, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);

        hStopButton = CreateWindowEx(0, L"BUTTON", L"Stop",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            100, 620, 80, 30,
            hwnd, (HMENU)ID_STOP_BUTTON, GetModuleHandle(NULL), NULL);

        hPauseButton = CreateWindowEx(0, L"BUTTON", L"Pause",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            190, 620, 80, 30,
            hwnd, (HMENU)ID_PAUSE_BUTTON, GetModuleHandle(NULL), NULL);

        hStepButton = CreateWindowEx(0, L"BUTTON", L"Step",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            280, 620, 80, 30,
            hwnd, (HMENU)ID_STEP_BUTTON, GetModuleHandle(NULL), NULL);

        hOutputArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL | ES_READONLY,
            620, 10, 560, 600,
            hwnd, (HMENU)ID_OUTPUT_AREA, GetModuleHandle(NULL), NULL);

        // Set timer for clock ticks
        SetTimer(hwnd, TIMER_ID, CLOCK_INTERVAL_MS, NULL);
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            // When the "Run" button is clicked
            char code[1024];
            GetWindowTextA(hTextArea, code, sizeof(code));
            run_vm(code, hOutputArea);
        }
        else if (LOWORD(wParam) == ID_STOP_BUTTON) {
            stop_vm(hOutputArea);
        }
        else if (LOWORD(wParam) == ID_PAUSE_BUTTON) {
            pause_vm(hOutputArea);
        }
        else if (LOWORD(wParam) == ID_STEP_BUTTON) {
            step_vm(hOutputArea);
        }
        break;

    case WM_TIMER:
        if (wParam == TIMER_ID && vm.running && !vm.paused) {
            // Clock signal tick
            OutputDebugStringA(std::to_string(clock_cycle).c_str());
            clock_cycle++;
            update_output(hOutputArea);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const wchar_t* CLASS_NAME = L"SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"Simple Virtual Machine",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 800,  // Resize to 1200x800
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif



// lepiej ale to jeszcze nie to
#if 0
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define ID_RUN_BUTTON 1
#define ID_TEXT_AREA  2
#define ID_OUTPUT_AREA 3
#define TIMER_ID 1

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024
#define CLOCK_INTERVAL_MS 500  // Interval for clock signal (500ms -> 2 Hz frequency)

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int instruction_pointer;
    int running;
} VM;

// Global VM
VM vm;

// Global output buffer
char output_buffer[2048];

// Clock simulation
int clock_cycle = 0;

// Line chart buffer
int clock_signal[100]; // Store clock signal history for visualization

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 1;
    ZeroMemory(output_buffer, sizeof(output_buffer));
}

// Function to append to the output buffer
void append_output(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsprintf(output_buffer + strlen(output_buffer), format, args);
    va_end(args);
}

// VM Instruction Set

// Arithmetic Instructions
void instr_add(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] + vm.registers[src2];
    append_output("ADD R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_sub(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] - vm.registers[src2];
    append_output("SUB R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_mul(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] * vm.registers[src2];
    append_output("MUL R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_div(int dest, int src1, int src2) {
    if (vm.registers[src2] == 0) {
        append_output("Error: Division by zero!\n");
        vm.running = 0;
        return;
    }
    vm.registers[dest] = vm.registers[src1] / vm.registers[src2];
    append_output("DIV R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

// Stack Operations
void instr_push(int reg) {
    if (vm.stack_ptr + 1 >= STACK_SIZE) {
        append_output("Stack Overflow!\n");
        vm.running = 0;
        return;
    }
    vm.stack[++vm.stack_ptr] = vm.registers[reg];
    append_output("PUSH R%d -> Stack[%d] = %d\n", reg, vm.stack_ptr, vm.registers[reg]);
}

void instr_pop(int reg) {
    if (vm.stack_ptr < 0) {
        append_output("Stack Underflow!\n");
        vm.running = 0;
        return;
    }
    vm.registers[reg] = vm.stack[vm.stack_ptr--];
    append_output("POP Stack[%d] -> R%d = %d\n", vm.stack_ptr + 1, reg, vm.registers[reg]);
}

// Branching
void instr_jmp(int addr) {
    vm.instruction_pointer = addr;
    append_output("JMP %d -> Instruction Pointer = %d\n", addr, addr);
}

void instr_jeq(int addr, int reg1, int reg2) {
    if (vm.registers[reg1] == vm.registers[reg2]) {
        vm.instruction_pointer = addr;
        append_output("JEQ -> JMP to %d (R%d == R%d)\n", addr, reg1, reg2);
    }
    else {
        append_output("JEQ -> No jump (R%d != R%d)\n", reg1, reg2);
    }
}

// Comparison Instructions
void instr_cmp(int reg1, int reg2) {
    int result = vm.registers[reg1] - vm.registers[reg2];
    append_output("CMP R%d, R%d -> Result = %d\n", reg1, reg2, result);
}

// Error message display for pseudo-compiler
void error_message(const wchar_t* message) {
    MessageBox(NULL, message, L"Compilation Error", MB_OK | MB_ICONERROR);
}

// Basic pseudo-compiler function with instruction size and line numbers
int compile_vm_code(const char* code) {
    char instruction[256];
    int pc = 0; // Program counter (instruction pointer)

    char* token = strtok((char*)code, "\n");
    int line_number = 1;

    while (token != NULL) {
        sscanf(token, "%s", instruction);
        append_output("Line %d: Address %d | ", line_number, pc);  // Show line number and address

        if (strcmp(instruction, "MOV") == 0) {
            int reg, val;
            if (sscanf(token, "MOV R%d, %d", &reg, &val) == 2) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    vm.registers[reg] = val;
                    append_output("MOV R%d, %d -> R%d = %d\n", reg, val, reg, val);
                }
                else {
                    error_message(L"Invalid register in MOV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MOV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "ADD") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "ADD R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_add(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in ADD instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid ADD syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "SUB") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "SUB R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_sub(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in SUB instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid SUB syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "MUL") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "MUL R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_mul(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in MUL instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MUL syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "DIV") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "DIV R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_div(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in DIV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid DIV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "CMP") == 0) {
            int reg1, reg2;
            if (sscanf(token, "CMP R%d, R%d", &reg1, &reg2) == 2) {
                if (reg1 >= 0 && reg1 < NUM_REGISTERS && reg2 >= 0 && reg2 < NUM_REGISTERS) {
                    instr_cmp(reg1, reg2);
                }
                else {
                    error_message(L"Invalid register in CMP instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid CMP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "PUSH") == 0) {
            int reg;
            if (sscanf(token, "PUSH R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_push(reg);
                }
                else {
                    error_message(L"Invalid register in PUSH instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid PUSH syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "POP") == 0) {
            int reg;
            if (sscanf(token, "POP R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_pop(reg);
                }
                else {
                    error_message(L"Invalid register in POP instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid POP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JMP") == 0) {
            int addr;
            if (sscanf(token, "JMP %d", &addr) == 1) {
                instr_jmp(addr);
            }
            else {
                error_message(L"Invalid JMP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JEQ") == 0) {
            int addr, reg1, reg2;
            if (sscanf(token, "JEQ %d, R%d, R%d", &addr, &reg1, &reg2) == 3) {
                instr_jeq(addr, reg1, reg2);
            }
            else {
                error_message(L"Invalid JEQ syntax.");
                return 0;
            }
        }
        else {
            wchar_t msg[256];
            swprintf(msg, 256, L"Unknown instruction: %S", instruction);
            error_message(msg);
            return 0;
        }

        token = strtok(NULL, "\n");
        pc += 4; // Assuming each instruction is 4 bytes in size
        line_number++;
    }

    return 1; // Compilation successful
}

// VM runner function with infinite loop
void run_vm(const char* code, HWND hOutputArea) {
    reset_vm();
    if (compile_vm_code(code)) {
        // Start VM execution in an infinite loop driven by the clock
        vm.running = 1;
        SetWindowTextA(hOutputArea, output_buffer);  // Display the initial results in the output area
    }
}

// Function to simulate clock ticks
void on_clock_tick(HWND hwnd) {
    if (vm.running) {
        clock_cycle++;  // Increment clock cycle

        // Add clock signal to history for drawing chart
        clock_signal[clock_cycle % 100] = 1;  // Simulate clock "high"

        if (clock_cycle % 2 == 0) {
            clock_signal[clock_cycle % 100] = 0;  // Simulate clock "low"
        }

        InvalidateRect(hwnd, NULL, FALSE);  // Force the window to repaint the chart
    }
}

// Draw clock chart using GDI
void draw_clock_chart(HDC hdc, RECT rect) {
    HPEN hPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255));
    SelectObject(hdc, hPen);

    // Draw the clock signal chart
    int prev_y = rect.bottom - 50;
    for (int i = 0; i < 100; i++) {
        int x = rect.left + i * 5;
        int y = (clock_signal[i] == 1) ? rect.bottom - 100 : rect.bottom - 50;
        MoveToEx(hdc, x, prev_y, NULL);
        LineTo(hdc, x, y);
        prev_y = y;
    }

    DeleteObject(hPen);
}

// Window procedure for handling messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hOutputArea, hButton;

    switch (uMsg) {
    case WM_CREATE:
        hTextArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL,
            10, 10, 600, 600,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        hButton = CreateWindowEx(0, L"BUTTON", L"Run",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 620, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);

        hOutputArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL | ES_READONLY,
            620, 10, 560, 600,
            hwnd, (HMENU)ID_OUTPUT_AREA, GetModuleHandle(NULL), NULL);

        // Set timer for clock ticks
        SetTimer(hwnd, TIMER_ID, CLOCK_INTERVAL_MS, NULL);
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            // When the "Run" button is clicked
            char code[1024];
            GetWindowTextA(hTextArea, code, sizeof(code));
            run_vm(code, hOutputArea);
        }
        break;

    case WM_TIMER:
        if (wParam == TIMER_ID) {
            on_clock_tick(hwnd);
        }
        break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT client_rect;
        GetClientRect(hwnd, &client_rect);

        // Draw clock signal chart at the bottom of the window
        RECT chart_rect = { 10, client_rect.bottom - 150, client_rect.right - 20, client_rect.bottom - 50 };
        draw_clock_chart(hdc, chart_rect);

        EndPaint(hwnd, &ps);
    }
    break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const wchar_t* CLASS_NAME = L"SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"Simple Virtual Machine",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 800,  // Resize to 1200x800
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif





// NO TO JU¯ JEST OK ALE MOZE BYC LEPIEJ
#if 0
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define ID_RUN_BUTTON 1
#define ID_TEXT_AREA  2
#define ID_OUTPUT_AREA 3

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int instruction_pointer;
    int running;
} VM;

// Global VM
VM vm;

// Global output buffer
char output_buffer[2048];

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 1;
    ZeroMemory(output_buffer, sizeof(output_buffer));
}

// Function to append to the output buffer
void append_output(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsprintf(output_buffer + strlen(output_buffer), format, args);
    va_end(args);
}

// VM Instruction Set

// Arithmetic Instructions
void instr_add(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] + vm.registers[src2];
    append_output("ADD R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_sub(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] - vm.registers[src2];
    append_output("SUB R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_mul(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] * vm.registers[src2];
    append_output("MUL R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

void instr_div(int dest, int src1, int src2) {
    if (vm.registers[src2] == 0) {
        append_output("Error: Division by zero!\n");
        vm.running = 0;
        return;
    }
    vm.registers[dest] = vm.registers[src1] / vm.registers[src2];
    append_output("DIV R%d, R%d, R%d -> R%d = %d\n", dest, src1, src2, dest, vm.registers[dest]);
}

// Stack Operations
void instr_push(int reg) {
    if (vm.stack_ptr + 1 >= STACK_SIZE) {
        append_output("Stack Overflow!\n");
        vm.running = 0;
        return;
    }
    vm.stack[++vm.stack_ptr] = vm.registers[reg];
    append_output("PUSH R%d -> Stack[%d] = %d\n", reg, vm.stack_ptr, vm.registers[reg]);
}

void instr_pop(int reg) {
    if (vm.stack_ptr < 0) {
        append_output("Stack Underflow!\n");
        vm.running = 0;
        return;
    }
    vm.registers[reg] = vm.stack[vm.stack_ptr--];
    append_output("POP Stack[%d] -> R%d = %d\n", vm.stack_ptr + 1, reg, vm.registers[reg]);
}

// Branching
void instr_jmp(int addr) {
    vm.instruction_pointer = addr;
    append_output("JMP %d -> Instruction Pointer = %d\n", addr, addr);
}

void instr_jeq(int addr, int reg1, int reg2) {
    if (vm.registers[reg1] == vm.registers[reg2]) {
        vm.instruction_pointer = addr;
        append_output("JEQ -> JMP to %d (R%d == R%d)\n", addr, reg1, reg2);
    }
    else {
        append_output("JEQ -> No jump (R%d != R%d)\n", reg1, reg2);
    }
}

// Comparison Instructions
void instr_cmp(int reg1, int reg2) {
    int result = vm.registers[reg1] - vm.registers[reg2];
    append_output("CMP R%d, R%d -> Result = %d\n", reg1, reg2, result);
}

// Error message display for pseudo-compiler
void error_message(const wchar_t* message) {
    MessageBox(NULL, message, L"Compilation Error", MB_OK | MB_ICONERROR);
}

// Basic pseudo-compiler function
int compile_vm_code(const char* code) {
    char instruction[256];
    int pc = 0; // Program counter (instruction pointer)

    char* token = strtok((char*)code, "\n");
    while (token != NULL) {
        sscanf(token, "%s", instruction);

        if (strcmp(instruction, "MOV") == 0) {
            int reg, val;
            if (sscanf(token, "MOV R%d, %d", &reg, &val) == 2) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    vm.registers[reg] = val;
                    append_output("MOV R%d, %d -> R%d = %d\n", reg, val, reg, val);
                }
                else {
                    error_message(L"Invalid register in MOV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MOV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "ADD") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "ADD R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_add(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in ADD instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid ADD syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "SUB") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "SUB R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_sub(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in SUB instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid SUB syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "MUL") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "MUL R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_mul(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in MUL instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MUL syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "DIV") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "DIV R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_div(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in DIV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid DIV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "CMP") == 0) {
            int reg1, reg2;
            if (sscanf(token, "CMP R%d, R%d", &reg1, &reg2) == 2) {
                if (reg1 >= 0 && reg1 < NUM_REGISTERS && reg2 >= 0 && reg2 < NUM_REGISTERS) {
                    instr_cmp(reg1, reg2);
                }
                else {
                    error_message(L"Invalid register in CMP instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid CMP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "PUSH") == 0) {
            int reg;
            if (sscanf(token, "PUSH R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_push(reg);
                }
                else {
                    error_message(L"Invalid register in PUSH instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid PUSH syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "POP") == 0) {
            int reg;
            if (sscanf(token, "POP R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_pop(reg);
                }
                else {
                    error_message(L"Invalid register in POP instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid POP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JMP") == 0) {
            int addr;
            if (sscanf(token, "JMP %d", &addr) == 1) {
                instr_jmp(addr);
            }
            else {
                error_message(L"Invalid JMP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JEQ") == 0) {
            int addr, reg1, reg2;
            if (sscanf(token, "JEQ %d, R%d, R%d", &addr, &reg1, &reg2) == 3) {
                instr_jeq(addr, reg1, reg2);
            }
            else {
                error_message(L"Invalid JEQ syntax.");
                return 0;
            }
        }
        else {
            wchar_t msg[256];
            swprintf(msg, 256, L"Unknown instruction: %S", instruction);
            error_message(msg);
            return 0;
        }

        token = strtok(NULL, "\n");
    }

    return 1; // Compilation successful
}

// VM runner function
void run_vm(const char* code, HWND hOutputArea) {
    reset_vm();
    if (compile_vm_code(code)) {
        // Execution can begin
        SetWindowTextA(hOutputArea, output_buffer);  // Display the results in the output area
    }
}

// Callback function for handling Windows messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hOutputArea, hButton;

    switch (uMsg) {
    case WM_CREATE:
        hTextArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL,
            10, 10, 600, 600,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        hButton = CreateWindowEx(0, L"BUTTON", L"Run",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 620, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);

        hOutputArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE | WS_VSCROLL | ES_READONLY,
            620, 10, 560, 600,
            hwnd, (HMENU)ID_OUTPUT_AREA, GetModuleHandle(NULL), NULL);
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            // When the "Run" button is clicked
            char code[1024];
            GetWindowTextA(hTextArea, code, sizeof(code));
            run_vm(code, hOutputArea);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const wchar_t* CLASS_NAME = L"SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"Simple Virtual Machine",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 800,  // Resize to 1200x800
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif




#if 0
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define ID_RUN_BUTTON 1
#define ID_TEXT_AREA  2

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int instruction_pointer;
    int running;
} VM;

// Global VM
VM vm;

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 1;
}

// VM Instruction Set

// Arithmetic Instructions
void instr_add(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] + vm.registers[src2];
}

void instr_sub(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] - vm.registers[src2];
}

// Stack Operations
void instr_push(int reg) {
    if (vm.stack_ptr + 1 >= STACK_SIZE) {
        MessageBox(NULL, L"Stack Overflow", L"VM Error", MB_OK | MB_ICONERROR);
        vm.running = 0;
        return;
    }
    vm.stack[++vm.stack_ptr] = vm.registers[reg];
}

void instr_pop(int reg) {
    if (vm.stack_ptr < 0) {
        MessageBox(NULL, L"Stack Underflow", L"VM Error", MB_OK | MB_ICONERROR);
        vm.running = 0;
        return;
    }
    vm.registers[reg] = vm.stack[vm.stack_ptr--];
}

// Branching
void instr_jmp(int addr) {
    vm.instruction_pointer = addr;
}

void instr_jeq(int addr, int reg1, int reg2) {
    if (vm.registers[reg1] == vm.registers[reg2]) {
        vm.instruction_pointer = addr;
    }
}

// Error message display for pseudo-compiler
void error_message(const wchar_t* message) {
    MessageBox(NULL, message, L"Compilation Error", MB_OK | MB_ICONERROR);
}

// Basic pseudo-compiler function
int compile_vm_code(const char* code) {
    char instruction[256];
    int pc = 0; // Program counter (instruction pointer)

    char* token = strtok((char*)code, "\n");
    while (token != NULL) {
        sscanf(token, "%s", instruction);

        if (strcmp(instruction, "MOV") == 0) {
            int reg, val;
            if (sscanf(token, "MOV R%d, %d", &reg, &val) == 2) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    vm.registers[reg] = val;
                }
                else {
                    error_message(L"Invalid register in MOV instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid MOV syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "ADD") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "ADD R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_add(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in ADD instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid ADD syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "SUB") == 0) {
            int dest, src1, src2;
            if (sscanf(token, "SUB R%d, R%d, R%d", &dest, &src1, &src2) == 3) {
                if (dest >= 0 && dest < NUM_REGISTERS && src1 >= 0 && src1 < NUM_REGISTERS && src2 >= 0 && src2 < NUM_REGISTERS) {
                    instr_sub(dest, src1, src2);
                }
                else {
                    error_message(L"Invalid register in SUB instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid SUB syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "PUSH") == 0) {
            int reg;
            if (sscanf(token, "PUSH R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_push(reg);
                }
                else {
                    error_message(L"Invalid register in PUSH instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid PUSH syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "POP") == 0) {
            int reg;
            if (sscanf(token, "POP R%d", &reg) == 1) {
                if (reg >= 0 && reg < NUM_REGISTERS) {
                    instr_pop(reg);
                }
                else {
                    error_message(L"Invalid register in POP instruction.");
                    return 0;
                }
            }
            else {
                error_message(L"Invalid POP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JMP") == 0) {
            int addr;
            if (sscanf(token, "JMP %d", &addr) == 1) {
                instr_jmp(addr);
            }
            else {
                error_message(L"Invalid JMP syntax.");
                return 0;
            }
        }
        else if (strcmp(instruction, "JEQ") == 0) {
            int addr, reg1, reg2;
            if (sscanf(token, "JEQ %d, R%d, R%d", &addr, &reg1, &reg2) == 3) {
                instr_jeq(addr, reg1, reg2);
            }
            else {
                error_message(L"Invalid JEQ syntax.");
                return 0;
            }
        }
        else {
            wchar_t msg[256];
            swprintf(msg, 256, L"Unknown instruction: %S", instruction);
            error_message(msg);
            return 0;
        }

        token = strtok(NULL, "\n");
    }

    return 1; // Compilation successful
}

// VM runner function
void run_vm(const char* code) {
    reset_vm();
    if (compile_vm_code(code)) {
        // Execution can begin
        // For testing, we'll just display the result in a message box
        wchar_t output[256];
        swprintf(output, 256, L"R0: %d, R1: %d, R2: %d, R3: %d", vm.registers[0], vm.registers[1], vm.registers[2], vm.registers[3]);
        MessageBox(NULL, output, L"VM Output", MB_OK);
    }
}

// Callback function for handling Windows messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hButton;

    switch (uMsg) {
    case WM_CREATE:
        hTextArea = CreateWindowEx(0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE,
            10, 10, 400, 200,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        hButton = CreateWindowEx(0, L"BUTTON", L"Run",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 220, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            // When the "Run" button is clicked
            char code[1024];
            GetWindowTextA(hTextArea, code, sizeof(code));
            run_vm(code);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const wchar_t* CLASS_NAME = L"SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"Simple Virtual Machine",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1200, 800,
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

#endif



#if 0
#include <windows.h>
#include <stdio.h>

#define ID_RUN_BUTTON 1
#define ID_TEXT_AREA  2

// Define Virtual Machine components
#define NUM_REGISTERS 10
#define STACK_SIZE 256
#define MEMORY_SIZE 1024

typedef struct {
    int registers[NUM_REGISTERS];
    int stack[STACK_SIZE];
    int stack_ptr;
    int memory[MEMORY_SIZE];
    int instruction_pointer;
    int running;
} VM;

// Global VM
VM vm;

// Function to reset VM state
void reset_vm() {
    ZeroMemory(vm.registers, sizeof(vm.registers));
    ZeroMemory(vm.stack, sizeof(vm.stack));
    ZeroMemory(vm.memory, sizeof(vm.memory));
    vm.stack_ptr = -1;
    vm.instruction_pointer = 0;
    vm.running = 1;
}

// VM Instruction Set

// Arithmetic Instructions
void instr_add(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] + vm.registers[src2];
}

void instr_sub(int dest, int src1, int src2) {
    vm.registers[dest] = vm.registers[src1] - vm.registers[src2];
}

// Stack Operations
void instr_push(int reg) {
    vm.stack[++vm.stack_ptr] = vm.registers[reg];
}

void instr_pop(int reg) {
    vm.registers[reg] = vm.stack[vm.stack_ptr--];
}

// Branching
void instr_jmp(int addr) {
    vm.instruction_pointer = addr;
}

void instr_jeq(int addr, int reg1, int reg2) {
    if (vm.registers[reg1] == vm.registers[reg2])
        vm.instruction_pointer = addr;
}

void run_vm(const char* code) {
    reset_vm();
    // Simple pseudo-compiler would go here, parsing code and running the VM
    // (e.g. converting instructions into opcodes and executing them)
    // For simplicity, let's assume the code is already "compiled"

    // For example, manually simulate code:
    // MOV R0, 10; MOV R1, 20; ADD R2, R0, R1; PUSH R2; POP R3;

    vm.registers[0] = 10;
    vm.registers[1] = 20;
    instr_add(2, 0, 1);
    instr_push(2);
    instr_pop(3);

    // Add more execution logic here...

    // For testing, we'll just display the result in a message box
    char output[256];
    sprintf(output, "R0: %d, R1: %d, R2: %d, R3: %d", vm.registers[0], vm.registers[1], vm.registers[2], vm.registers[3]);
    MessageBox(NULL, (LPCWSTR)output, L"VM Output", MB_OK);
}

// Callback function for handling Windows messages
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hTextArea, hButton;

    switch (uMsg) {
    case WM_CREATE:
        hTextArea = CreateWindowEx(0, TEXT("EDIT"), TEXT(""),
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_MULTILINE,
            10, 10, 400, 200,
            hwnd, (HMENU)ID_TEXT_AREA, GetModuleHandle(NULL), NULL);

        hButton = CreateWindowEx(0, TEXT("BUTTON"), TEXT("Run"),
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            10, 220, 80, 30,
            hwnd, (HMENU)ID_RUN_BUTTON, GetModuleHandle(NULL), NULL);
        break;

    case WM_COMMAND:
        if (LOWORD(wParam) == ID_RUN_BUTTON) {
            // When the "Run" button is clicked
            char code[1024];
            GetWindowText(hTextArea, (LPWSTR)code, sizeof(code));
            run_vm(code);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// WinMain: Entry point for a Windows application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    const char* CLASS_NAME = "SimpleVMWindowClass";
    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCWSTR)CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, (LPCWSTR)CLASS_NAME, TEXT("Simple Virtual Machine"),
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 600, 400,
        NULL, NULL, hInstance, NULL);

    if (!hwnd) {
        return 0;
    }

    // Main message loop
    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
#endif



#if 0
// simple_virtual_machine.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "simple_virtual_machine.h"

#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Place code here.

    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_SIMPLEVIRTUALMACHINE, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_SIMPLEVIRTUALMACHINE));

    MSG msg;

    // Main message loop:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_SIMPLEVIRTUALMACHINE));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_SIMPLEVIRTUALMACHINE);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Store instance handle in our global variable

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // Parse the menu selections:
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code that uses hdc here...
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
#endif