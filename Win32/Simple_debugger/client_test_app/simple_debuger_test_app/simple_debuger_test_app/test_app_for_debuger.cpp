#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cause_access_violation() {
    int* ptr = NULL;
    printf("Causing Access Violation...\n");
    *ptr = 42;  // Dereferencing a NULL pointer
}

void cause_buffer_overflow() {
    char buffer[8];
    printf("Causing Buffer Overflow...\n");
    strcpy(buffer, "This is a very long string that will overflow the buffer!"); // Buffer overflow
}

void cause_divide_by_zero() {
    int a = 1;
    int b = 0;
    printf("Causing Divide by Zero...\n");
    int c = a / b;  // Division by zero
    printf("Result: %d\n", c);
}

void normal_loop()
{
    int x = 0;
    while (1) {
        printf("%d \n", x);
        x += 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <exception_type>\n", argv[0]);
        printf("Exception Types: \n");
        printf("1 - Access Violation\n");
        printf("2 - Buffer Overflow\n");
        printf("3 - Divide by Zero\n");
        //return 1;
    }



    int c;
    while ((c = getchar()) != EOF) {

        //int choice = (choice = atoi(c))
        int choice = c;

        printf("%d \n", choice);

            switch (choice) {
            case 49:
                cause_access_violation();
                break;
            case 50:
                cause_buffer_overflow();
                break;
            case 51:
                cause_divide_by_zero();
                break;
            case 52:
                normal_loop();
                break;
            default:
                printf("Invalid choice.\n");
                break;
            }
    }

    printf("Program finished without crashing!\n");
    return 0;
}
