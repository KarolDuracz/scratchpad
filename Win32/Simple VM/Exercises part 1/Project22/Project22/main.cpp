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

        // Switch to Fiber 2 after completing
        SwitchToFiber(fiber[1]);
    }
}

void FiberFunction2(void* param) {
    while (1) {
        HANDLE* events = (HANDLE*)param;
        DWORD dwWaitResult = WaitForMultipleObjects(1, &events[1], FALSE, INFINITE);

        if (dwWaitResult == WAIT_OBJECT_0) {
            std::cout << "Fiber 2 received event2 and proceeding." << std::endl;
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
