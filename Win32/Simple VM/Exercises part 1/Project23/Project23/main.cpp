#include <windows.h>
#include <iostream>

#define MAX_QUEUE_SIZE 20

typedef struct {
    HANDLE events[2];
    HANDLE fiber[MAX_QUEUE_SIZE];
} FIBER_DATA;

HANDLE fiber[MAX_QUEUE_SIZE];

void FiberFunction1(void* param) {
    int x = 0;
    while (1) {
        x += 1;
        HANDLE* events = (HANDLE*)param;
        DWORD dwWaitResult = WaitForMultipleObjects(1, &events[0], FALSE, INFINITE);

        if (dwWaitResult == WAIT_OBJECT_0) {
            std::cout << "Fiber 1 received event1 and proceeding." << " [ " << x << " ]" << std::endl;
        }

        std::cout << param << " " << events[1] << std::endl;

        
        if (x > 10) {
            // Switch to Fiber 2 after completing
            x = 0;
            SwitchToFiber(fiber[1]);
            
        }
    }
}

void FiberFunction2(void* param) {
    while (1) {
        HANDLE* events = (HANDLE*)param;
        DWORD dwWaitResult = WaitForMultipleObjects(1, &events[1], FALSE, INFINITE);

        if (dwWaitResult == WAIT_OBJECT_0) {
            std::cout << "Fiber >>>>> 2 <<<<< received event2 and proceeding." << std::endl;
        }

        // Switch back to Fiber 1 after completing
        SwitchToFiber(fiber[0]);
    }
}

DWORD WINAPI SignalEvents(LPVOID param) {
    while (1) {
        HANDLE* events = (HANDLE*)param;

        // Signal event1 to start Fiber 1
        SetEvent(events[0]);
        std::cout << "Event1 signaled to start Fiber 1." << std::endl;

        // Sleep for a while before signaling event2 to start Fiber 2
        Sleep(500); // Simulate some work before signaling event2
        SetEvent(events[1]);
        std::cout << "Event2 signaled to start Fiber 2." << std::endl;
    }
    return 0;
}

int main() {
    // Convert the main thread to a fiber
    void* mainFiber = ConvertThreadToFiber(NULL);

    FIBER_DATA fq;

    // Create two synchronization events
    HANDLE event1 = CreateEvent(NULL, FALSE, FALSE, NULL);  // Fiber 1 event
    HANDLE event2 = CreateEvent(NULL, FALSE, FALSE, NULL);  // Fiber 2 event

    // Prepare the events array to pass to fibers
    HANDLE events[2] = { event1, event2 };

    std::cout << " print id of events " << event1 << " " << event2 << std::endl;

    // Create two fibers
    void* fiber1 = CreateFiber(0, (LPFIBER_START_ROUTINE)FiberFunction1, (void*)events);
    void* fiber2 = CreateFiber(0, (LPFIBER_START_ROUTINE)FiberFunction2, (void*)events);

    std::cout << "name of fibers " << fiber1 << " " << fiber2 << std::endl;

    fiber[0] = fiber1;
    fiber[1] = fiber2;

    // Create a thread to signal the events
    HANDLE signalThread = CreateThread(NULL, 0, SignalEvents, (LPVOID)events, 0, NULL);

    // Start Fiber 1, it will wait on event1
    SwitchToFiber(fiber1);

    // Wait for the signaling thread to finish
    WaitForSingleObject(signalThread, INFINITE);

    // Fiber 1 and Fiber 2 should have completed by now, so clean up
    DeleteFiber(fiber1);
    DeleteFiber(fiber2);
    CloseHandle(event1);
    CloseHandle(event2);
    CloseHandle(signalThread);

    // Convert the main fiber back to a regular thread
    ConvertFiberToThread();

    return 0;
}


#if 0
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// Fiber structure to hold context
typedef struct Fiber {
    void* stack;          // Pointer to allocated stack memory
    void* stackPointer;   // Current stack pointer
    void (*function)(void*); // Fiber function to execute
    void* param;          // Parameter to the fiber function
} Fiber;

// Pointers to the current and next fibers
static Fiber* currentFiber = NULL;
static Fiber* nextFiber = NULL;

#if 0
// Function to save and switch context (inline assembly for x86)
void SwitchContext(Fiber* from, Fiber* to) {
    static void* savedFromStackPointer = NULL;
    static void* savedToStackPointer = NULL;

    // Save stack pointers into static variables
    savedFromStackPointer = from->stackPointer;
    savedToStackPointer = to->stackPointer;

    __asm {
        // Save registers of the current fiber
        mov savedFromStackPointer, esp  // Save the current stack pointer
        mov eax, from
        mov[eax], ebx                 // Save EBX
        mov[eax + 4], esi               // Save ESI
        mov[eax + 8], edi               // Save EDI

        // Load registers of the next fiber
        mov eax, to
        mov ebx, [eax]                 // Restore EBX
        mov esp, savedToStackPointer   // Restore stack pointer
        mov esi, [eax + 4]               // Restore ESI
        mov edi, [eax + 8]               // Restore EDI
    }
}
#endif

__declspec(naked) void SwitchContext(Fiber* from, Fiber* to) {
    __asm {
        // Save the context of the "from" fiber
        mov eax, [esp + 4]      // Load 'from' pointer (first parameter)
        test eax, eax           // Check if 'from' is NULL
        jz skip_save            // If NULL, skip saving context

        mov[eax + 0], ebx      // Save EBX
        mov[eax + 4], esi      // Save ESI
        mov[eax + 8], edi      // Save EDI
        mov[eax + 12], esp     // Save ESP (stack pointer)

        skip_save :
        // Load the context of the "to" fiber
        mov eax, [esp + 8]      // Load 'to' pointer (second parameter)
        mov ebx, [eax + 0]      // Restore EBX
        mov esi, [eax + 4]      // Restore ESI
        mov edi, [eax + 8]      // Restore EDI

        mov ecx, [esp]
        //mov [eax + 0xc], ecx


        mov esp, [eax + 12]     // Restore ESP (stack pointer)
        push ecx

        ret                     // Return to the restored context
    }
}

// Function to start a fiber (called on first switch)
void FiberStart() {
    if (currentFiber && currentFiber->function) {
        currentFiber->function(currentFiber->param);
    }

    // Exit fiber after execution
    printf("Fiber has completed execution\n");
    ExitThread(0); // End the thread
}

// Function to create a new fiber
Fiber* CreateFiberRaw(size_t stackSize, void (*function)(void*), void* param) {
    Fiber* fiber = (Fiber*)malloc(sizeof(Fiber));
    if (!fiber) return NULL;

    // Allocate stack memory
    fiber->stack = malloc(stackSize);
    if (!fiber->stack) {
        free(fiber);
        return NULL;
    }

    // Set up the initial stack context
    void** stackTop = (void**)((char*)fiber->stack + stackSize);
    *(--stackTop) = (void*)param;      // Function parameter
    *(--stackTop) = NULL;              // Return address (dummy)
    *(--stackTop) = (void*)FiberStart; // Entry point of the fiber

    fiber->stackPointer = stackTop;
    fiber->function = function;
    fiber->param = param;

    return fiber;
}

// Function to delete a fiber
void DeleteFiberRaw(Fiber* fiber) {
    if (fiber) {
        if (fiber->stack) {
            free(fiber->stack);
        }
        free(fiber);
    }
}

// Example fiber functions
void FiberFunction1(void* param) {
    int* counter = (int*)param;
    while (1) {
        printf("Fiber 1: Counter = %d\n", (*counter)++);
        SwitchContext(currentFiber, nextFiber); // Yield to next fiber
    }
}

void FiberFunction2(void* param) {
    int* counter = (int*)param;
    while (1) {
        printf("Fiber 2: Counter = %d\n", (*counter)++);
        SwitchContext(currentFiber, nextFiber); // Yield to next fiber
    }
}

int main() {
    int counter1 = 0, counter2 = 100;

    // Create two fibers
    Fiber fiber1, fiber2;
    fiber1 = *CreateFiberRaw(1024 * 64, FiberFunction1, &counter1);
    fiber2 = *CreateFiberRaw(1024 * 64, FiberFunction2, &counter2);

    // Initialize current and next fibers
    currentFiber = &fiber1;
    nextFiber = &fiber2;

    std::cout << fiber1.stackPointer << " " << &fiber1 << std::endl;
    std::cout << currentFiber->function << std::endl;

    // Start fiber 1
    SwitchContext(NULL, currentFiber);

    // Cleanup
    DeleteFiberRaw(&fiber1);
    DeleteFiberRaw(&fiber2);

    return 0;
}
#endif
