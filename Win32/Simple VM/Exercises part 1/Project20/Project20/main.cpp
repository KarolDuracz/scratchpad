#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <stdbool.h>
#include <math.h>

#define REGISTER_COUNT 8
#define MEMORY_SIZE 16
#define STACK_SIZE 16
#define CODE_SIZE 256
#define INPUT_BUFFER_SIZE 128

// Opcodes
typedef enum {
    NOP = 0,
    ADD,
    SUB,
    MUL,
    DIV,
    CMP,
    JMP,
    CALL,
    RET,
    PUSH,
    POP,
    LOAD_XW,
    TANH,
    HALT
} Opcode;

// Instruction structure
typedef struct {
    Opcode opcode;
    int operand1;
    int operand2;
    int operand3;
} Instruction;

// Task context
typedef struct {
    unsigned long long registers[REGISTER_COUNT];
    unsigned long long memory[MEMORY_SIZE];
    unsigned long long stack[STACK_SIZE];
    Instruction code[CODE_SIZE];
    unsigned long long rip;
    int sp;
    bool is_complete;
} TaskContext;

// Control flags and handles
volatile bool is_running = false;
HANDLE consoleEvent;
HANDLE globalInterruptEvent;

// Function prototypes
void printTaskState(TaskContext* context);
void simulateInstruction(TaskContext* context);
double tanhFunction(double x);
bool parseInstruction(const char* input, Instruction* instr);
void executePrompt(TaskContext* context);
DWORD WINAPI interruptHandler(LPVOID param);
DWORD WINAPI globalInterruptHandler(LPVOID param);

// Function to print task state
void printTaskState(TaskContext* context) {
    printf("Current State:\n");
    printf("RIP: %llu\n", context->rip);
    for (int i = 0; i < REGISTER_COUNT; i++) {
        printf("R%d: %llu\n", i, context->registers[i]);
    }
    printf("\n");
}

// Tanh function
double tanhFunction(double x) {
    return tanh(x);
}

// Simulate instruction
void simulateInstruction(TaskContext* context, unsigned long long value1, unsigned long long value2) {
    if (context->is_complete) return;

    Instruction* instr = &context->code[context->rip];

    switch (instr->opcode) {
    case LOAD_XW:
        if (value1 != 0 || value2 != 0) { // If extra values are provided
            context->registers[instr->operand1] = value1;
            context->registers[instr->operand2] = value2;
        }
        context->registers[instr->operand3] =
            context->registers[instr->operand1] *
            context->registers[instr->operand2];
        context->rip++;
        break;
        // Other cases remain unchanged...
    case ADD:
        context->registers[instr->operand3] =
            context->registers[instr->operand1] +
            context->registers[instr->operand2];
        context->rip++;
        break;
    case HALT:
        context->is_complete = true;
        break;
    default:
        printf("Unknown instruction!\n");
        context->is_complete = true;
        break;
    }
}

#if 0
// Simulate instruction
void simulateInstruction(TaskContext* context) {
    if (context->is_complete) return;



    Instruction* instr = &context->code[context->rip];


    printf("%d %d \n", context->registers[0], instr->operand1);

    switch (instr->opcode) {
    case NOP:
        context->rip++;
        break;
    case ADD:
        context->registers[instr->operand3] =
            context->registers[instr->operand1] +
            context->registers[instr->operand2];

        printf("result of add %d %d %d %d %d %d \n", context->registers[instr->operand3], instr->operand3, instr->operand1, instr->operand2,
            context->registers[instr->operand1], context->registers[instr->operand2]);
        for (int i = 0; i < REGISTER_COUNT; i++) {
            printf("[%d]:%d ", i, context->registers[i]);
        }

        context->rip++;
        break;
    case SUB:
        context->registers[instr->operand3] =
            context->registers[instr->operand1] -
            context->registers[instr->operand2];
        context->rip++;
        break;
    case MUL:
        context->registers[instr->operand3] =
            context->registers[instr->operand1] *
            context->registers[instr->operand2];
        context->rip++;
        break;
    case DIV:
        context->registers[instr->operand3] =
            context->registers[instr->operand1] /
            context->registers[instr->operand2];
        context->rip++;
        break;
    case CMP:
        context->registers[0] =
            (context->registers[instr->operand1] ==
                context->registers[instr->operand2])
            ? 1
            : 0;
        context->rip++;
        break;
    case JMP:
        context->rip = instr->operand1;
        break;
    case CALL:
        context->stack[++context->sp] = context->rip + 1;
        context->rip = instr->operand1;
        break;
    case RET:
        context->rip = context->stack[context->sp--];
        break;
    case PUSH:
        context->stack[++context->sp] = context->registers[instr->operand1];
        context->rip++;
        break;
    case POP:
        context->registers[instr->operand1] = context->stack[context->sp--];
        context->rip++;
        break;
    case LOAD_XW:
        context->registers[instr->operand3] =
            context->registers[instr->operand1] *
            context->registers[instr->operand2];
        context->rip++;
        break;
    case TANH:
        context->registers[instr->operand3] =
            (unsigned long long)tanhFunction(
                (double)context->registers[instr->operand1]);
        context->rip++;
        break;
    case HALT:
        context->is_complete = true;
        break;
    default:
        printf("Unknown instruction!\n");
        context->is_complete = true;
        break;
    }
}
#endif


// Parse user input into an instruction
bool parseInstruction(const char* input, Instruction* instr, unsigned long long* value1, unsigned long long* value2) {
    char opcode_str[16];
    int op1, op2, op3;
    *value1 = 0;  // Default values
    *value2 = 0;

    // Check for extended format {LOAD_XW, 0, 1, 3, {value1, value2}}
    if (sscanf(input, "{%[^,], %d, %d, %d, {%llu, %llu}}", opcode_str, &op1, &op2, &op3, value1, value2) == 6) {
        printf("Extended format detected: {%llu, %llu}\n", *value1, *value2);
    }
    else if (sscanf(input, "{%[^,], %d, %d, %d}", opcode_str, &op1, &op2, &op3) != 4) {
        printf("Invalid instruction format. Use {OPCODE, op1, op2, op3} or {OPCODE, op1, op2, op3, {value1, value2}}\n");
        return false;
    }

    // Map opcode string to enum
    if (strcmp(opcode_str, "ADD") == 0) instr->opcode = ADD;
    else if (strcmp(opcode_str, "SUB") == 0) instr->opcode = SUB;
    else if (strcmp(opcode_str, "MUL") == 0) instr->opcode = MUL;
    else if (strcmp(opcode_str, "DIV") == 0) instr->opcode = DIV;
    else if (strcmp(opcode_str, "CMP") == 0) instr->opcode = CMP;
    else if (strcmp(opcode_str, "JMP") == 0) instr->opcode = JMP;
    else if (strcmp(opcode_str, "CALL") == 0) instr->opcode = CALL;
    else if (strcmp(opcode_str, "RET") == 0) instr->opcode = RET;
    else if (strcmp(opcode_str, "PUSH") == 0) instr->opcode = PUSH;
    else if (strcmp(opcode_str, "POP") == 0) instr->opcode = POP;
    else if (strcmp(opcode_str, "LOAD_XW") == 0) instr->opcode = LOAD_XW;
    else if (strcmp(opcode_str, "TANH") == 0) instr->opcode = TANH;
    else if (strcmp(opcode_str, "HALT") == 0) instr->opcode = HALT;
    else {
        printf("Unknown opcode: %s\n", opcode_str);
        return false;
    }

    instr->operand1 = op1;
    instr->operand2 = op2;
    instr->operand3 = op3;
    return true;
}

#if 0
// Parse user input into an instruction
bool parseInstruction(const char* input, Instruction* instr) {
    char opcode_str[16];
    int op1, op2, op3;
    if (sscanf(input, "{%[^,], %d, %d, %d}", opcode_str, &op1, &op2, &op3) != 4) {
        printf("Invalid instruction format. Use {OPCODE, op1, op2, op3}\n");
        return false;
    }

    // Map opcode string to enum
    if (strcmp(opcode_str, "ADD") == 0) instr->opcode = ADD;
    else if (strcmp(opcode_str, "SUB") == 0) instr->opcode = SUB;
    else if (strcmp(opcode_str, "MUL") == 0) instr->opcode = MUL;
    else if (strcmp(opcode_str, "DIV") == 0) instr->opcode = DIV;
    else if (strcmp(opcode_str, "CMP") == 0) instr->opcode = CMP;
    else if (strcmp(opcode_str, "JMP") == 0) instr->opcode = JMP;
    else if (strcmp(opcode_str, "CALL") == 0) instr->opcode = CALL;
    else if (strcmp(opcode_str, "RET") == 0) instr->opcode = RET;
    else if (strcmp(opcode_str, "PUSH") == 0) instr->opcode = PUSH;
    else if (strcmp(opcode_str, "POP") == 0) instr->opcode = POP;
    else if (strcmp(opcode_str, "LOAD_XW") == 0) instr->opcode = LOAD_XW;
    else if (strcmp(opcode_str, "TANH") == 0) instr->opcode = TANH;
    else if (strcmp(opcode_str, "HALT") == 0) instr->opcode = HALT;
    else {
        printf("Unknown opcode: %s\n", opcode_str);
        return false;
    }

    instr->operand1 = op1;
    instr->operand2 = op2;
    instr->operand3 = op3;

    printf("OPPERANDS -> %d %d %d \n", op1, op2, op3);

    return true;
}
#endif

// Execute instructions from prompt
void executePrompt(TaskContext* context) {
    char input[INPUT_BUFFER_SIZE];
    Instruction instr;
    unsigned long long value1, value2;

    while (!context->is_complete) {
        printf("Enter instruction: ");
        fgets(input, sizeof(input), stdin);

        if (parseInstruction(input, &instr, &value1, &value2)) {
            context->code[context->rip] = instr;
            SetEvent(consoleEvent);  // Signal console event
            simulateInstruction(context, value1, value2);  // Pass values
            printTaskState(context);
        }
    }
}

#if 0
// Execute instructions from prompt
void executePrompt(TaskContext* context) {
    char input[INPUT_BUFFER_SIZE];
    Instruction instr;

    while (!context->is_complete) {
        printf("Enter instruction: ");
        fgets(input, sizeof(input), stdin);

        printf("context->code %d \n", context->code[0]);

        if (parseInstruction(input, &instr)) {
            context->code[context->rip] = instr;

            printf("context->code %d %d %d %d \n", context->rip, context->code[0], instr.operand1, context->registers[0]);
            printf("context->code %d %d %d \n", context->rip, context->code[1], instr.operand1);

            // Signal console event to interrupt handler
            SetEvent(consoleEvent);

            simulateInstruction(context);
            printTaskState(context);
        }
    }
}
#endif

// Interrupt handler for console commands
DWORD WINAPI interruptHandler(LPVOID param) {
    while (is_running) {
        WaitForSingleObject(consoleEvent, INFINITE);
        printf("** Interrupt: Console command received **\n");
        ResetEvent(consoleEvent);
    }
    return 0;
}

// Global interrupt handler
DWORD WINAPI globalInterruptHandler(LPVOID param) {
    while (is_running) {
        WaitForSingleObject(globalInterruptEvent, INFINITE);
        printf("** Global Interrupt: System event **\n");
        ResetEvent(globalInterruptEvent);
    }
    return 0;
}

int main() {
    TaskContext context = { 0 };
    context.rip = 0;
    context.sp = -1;
    context.is_complete = false;

    context.registers[0] = 2; // X
    context.registers[1] = 3; // W
    context.registers[2] = 1; // B

    // Create events
    consoleEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    globalInterruptEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    // Start interrupt threads
    is_running = true;
    HANDLE consoleThread = CreateThread(NULL, 0, interruptHandler, NULL, 0, NULL);
    HANDLE globalThread = CreateThread(NULL, 0, globalInterruptHandler, NULL, 0, NULL);

    // Run instruction prompt
    executePrompt(&context);

    is_running = false;
    SetEvent(globalInterruptEvent); // Wake up threads to terminate
    SetEvent(consoleEvent);
    WaitForSingleObject(consoleThread, INFINITE);
    WaitForSingleObject(globalThread, INFINITE);

    CloseHandle(consoleThread);
    CloseHandle(globalThread);
    CloseHandle(consoleEvent);
    CloseHandle(globalInterruptEvent);

    return 0;
}
