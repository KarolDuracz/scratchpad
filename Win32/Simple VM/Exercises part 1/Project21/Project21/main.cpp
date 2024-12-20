#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <conio.h>

#define TEST_ITERATIONS 10
#define CPU_CLOCK_CALIBRATION_COUNT 1000000
#define TASK_COUNT 3
#define TIME_LIMIT 0.002  // Time limit for each task in seconds
#define SLEEP_TIME_MS 10  // Sleep time for each task in milliseconds

//#define PRINT_SWTICH_INFO

// Enum to represent task states
typedef enum {
    RUNNING,
    SLEEPING,
    COMPLETED
} TaskState;

// Structure to represent a task
typedef struct {
    int task_id;
    TaskState state;
    double start_time;
    double execution_time;
    HANDLE thread_handle;
    DWORD thread_id;
    CONTEXT thread_context;  // Store the thread context here
} Task;

// Structure to represent a clocked CPU mechanism
typedef struct {
    double base_frequency_mhz;  // Base CPU frequency in MHz
    double current_frequency;   // Current scaled frequency
    double tick_interval;       // Tick interval based on frequency
} CPUClock;

// Global scheduler for managing tasks
typedef struct {
    Task tasks[TASK_COUNT];
    int current_task_index;
    int total_tasks_completed;
} Scheduler;

// Function to get the current high-resolution time in seconds
double getCurrentTime() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

// Function to measure the base frequency in MHz
double measureBaseFrequency() {
    double durations[TEST_ITERATIONS];
    double total_time = 0.0;

    for (int i = 0; i < TEST_ITERATIONS; i++) {
        double start = getCurrentTime();

        // Perform calibration task
        volatile int x = 0;
        for (int j = 0; j < CPU_CLOCK_CALIBRATION_COUNT; j++) {
            x += j;
        }

        double end = getCurrentTime();
        double elapsed = end - start;
        durations[i] = elapsed;
        total_time += elapsed;
    }

    double mean_time = total_time / TEST_ITERATIONS;
    return (CPU_CLOCK_CALIBRATION_COUNT / mean_time) / 1e6;
}

// Task execution function with infinite loop simulation
DWORD WINAPI taskExecution(LPVOID param) {
    Task* task = (Task*)param;
    task->start_time = getCurrentTime();  // Start time for the task execution

    while (1) {
        // Simulate task workload by sleeping for a while
        Sleep(SLEEP_TIME_MS);  // Simulate work

        // Update the execution time
        task->execution_time = getCurrentTime() - task->start_time;

        // Check if the task has exceeded the time limit
        if (task->execution_time > TIME_LIMIT) {
#ifdef PRINT_SWTICH_INFO
            printf("Task %d exceeded time limit (%.6f seconds). Switching.\n", task->task_id, task->execution_time);
#endif

            // Yield control to the scheduler by suspending this task
            task->state = SLEEPING;  // Set the state to sleeping, as it will be resumed later

            return 0;  // End the current execution but leave the task in the scheduler
        }
    }

    return 0;
}

// Scheduler function to manage task execution and time constraints
void schedulerExecute(Scheduler* scheduler) {
    while (scheduler->total_tasks_completed < TASK_COUNT) {
        Task* current_task = &scheduler->tasks[scheduler->current_task_index];

        if (current_task->state == RUNNING) {
            // Wait for the task to finish or exceed the time limit
            WaitForSingleObject(current_task->thread_handle, INFINITE);

            // Once the task completes or is switched, move to the next task
            scheduler->current_task_index = (scheduler->current_task_index + 1) % TASK_COUNT;

            // Check if the task is completed and increment the completed count
            if (current_task->state == COMPLETED) {
                scheduler->total_tasks_completed++;
            }
        }
        else if (current_task->state == SLEEPING) {
            // If the task is in SLEEPING state, resume it
#ifdef PRINT_SWTICH_INFO
            printf("Resuming Task %d.\n", current_task->task_id);
#endif
            current_task->state = RUNNING;
            current_task->start_time = getCurrentTime();  // Reset start time when resuming

            // Restart the task thread
            current_task->thread_handle = CreateThread(
                NULL,
                0,
                taskExecution,
                current_task,
                0,
                &current_task->thread_id
            );
        }

        // Yielding control to allow the next task to execute
        Sleep(1);  // Small sleep to give control back to the system
    }
}

// Initialize tasks and set up the scheduler
void initializeScheduler(Scheduler* scheduler) {
    for (int i = 0; i < TASK_COUNT; i++) {
        scheduler->tasks[i].task_id = i + 1;
        scheduler->tasks[i].state = RUNNING;
        scheduler->tasks[i].execution_time = 0.0;
        scheduler->tasks[i].thread_handle = NULL;
        scheduler->tasks[i].thread_id = 0;
    }
    scheduler->current_task_index = 0;
    scheduler->total_tasks_completed = 0;
}

// Function to print thread context information in the second console
void printThreadContext(Scheduler* scheduler) {
    while (1) {
        for (int i = 0; i < TASK_COUNT; i++) {
            Task* task = &scheduler->tasks[i];
            task->thread_context.ContextFlags = CONTEXT_FULL;
            SuspendThread(task->thread_handle);
            if (task->state == RUNNING || task->state == SLEEPING) {
                // Retrieve thread context
                
                if (GetThreadContext(task->thread_handle, &task->thread_context)) {
                    printf("Task %d (Thread ID: %lu): Context - EIP: %p, ESP: %p\n",
                        task->task_id,
                        task->thread_id,
                        (void*)task->thread_context.Eip,
                        (void*)task->thread_context.Esp);
                }
                else {
                    printf("Failed to get context for Task %d\n", task->task_id);
                    //ResumeThread(task->thread_handle);
                    printf("STATE %d \n", task->state);
                }
            }
            ResumeThread(task->thread_handle);
        }

        

        // Sleep for a while to avoid overwhelming the console with output
        Sleep(500); // Update context info every 500ms
    }
}

// Main execution function to simulate tasks running
void executeTasks() {
    Scheduler scheduler;
    initializeScheduler(&scheduler);

    printf("Starting task execution with global scheduler...\n");

    // Create threads for each task
    for (int i = 0; i < TASK_COUNT; i++) {
        scheduler.tasks[i].thread_handle = CreateThread(
            NULL,
            0,
            taskExecution,
            &scheduler.tasks[i],
            0,
            &scheduler.tasks[i].thread_id
        );
    }

    // Create a thread to manage scheduler and print context
    CreateThread(
        NULL,
        0,
        (LPTHREAD_START_ROUTINE)printThreadContext,
        &scheduler,
        0,
        NULL
    );

    // Execute tasks with the global scheduler
    schedulerExecute(&scheduler);
    printf("All tasks completed.\n");
}

int main() {
    // Step 1: Measure base CPU frequency
    printf("Measuring base frequency...\n");
    double base_frequency = measureBaseFrequency();
    printf("Measured base CPU frequency: %.2f MHz\n", base_frequency);

    // Step 2: Initialize the CPU clock
    CPUClock cpuClock;
    cpuClock.base_frequency_mhz = base_frequency;
    cpuClock.current_frequency = base_frequency;
    cpuClock.tick_interval = 1.0 / (base_frequency * 1e6);

    // Step 3: Execute tasks using global scheduler
    double start = getCurrentTime();
    executeTasks();
    double end = getCurrentTime();
    printf("Time taken for all tasks: %.6f seconds\n", end - start);

    return 0;
}
