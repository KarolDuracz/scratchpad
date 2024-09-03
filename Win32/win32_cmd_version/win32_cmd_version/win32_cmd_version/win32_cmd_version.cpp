#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <tchar.h>
#include <string.h>
#include <commctrl.h>

#pragma comment(lib, "Comctl32.lib")

#define IDC_MAIN_EDIT 101   // Edit control identifier
#define IDC_MAIN_TEXT 102   // Text area identifier
#define MAX_LINES 20        // Maximum number of visible lines
#define MAX_BUFFER_SIZE 256 // Maximum size per line

TCHAR lineBuffer[MAX_LINES][MAX_BUFFER_SIZE]; // Buffer for storing individual lines
int lineCount = 0;                            // Total number of lines entered

// Function to update the visible text area with the last MAX_LINES lines
void UpdateTextArea(HWND hText) {
    TCHAR displayBuffer[MAX_BUFFER_SIZE * MAX_LINES] = TEXT(""); // Buffer to accumulate the text for display

    for (int i = 0; i < min(lineCount, MAX_LINES); i++) {
        _tcscat(displayBuffer, lineBuffer[i]);
        _tcscat(displayBuffer, TEXT("\r\n"));
    }

    SetWindowText(hText, displayBuffer);
}

LRESULT CALLBACK EditProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam, UINT_PTR uIdSubclass, DWORD_PTR dwRefData) {
    if (uMsg == WM_KEYDOWN) {
        if (wParam == VK_RETURN) {
            HWND hText = (HWND)dwRefData;
            TCHAR textBuffer[MAX_BUFFER_SIZE];

            // Get the text from the edit control
            GetWindowText(hwnd, textBuffer, MAX_BUFFER_SIZE);

            // If the buffer is full, scroll up (move lines up by one)
            if (lineCount >= MAX_LINES) {
                for (int i = 0; i < MAX_LINES - 1; i++) {
                    _tcscpy(lineBuffer[i], lineBuffer[i + 1]);
                }
                _tcscpy(lineBuffer[MAX_LINES - 1], textBuffer);
            }
            else {
                _tcscpy(lineBuffer[lineCount], textBuffer);
                lineCount++;
            }

            // Update the text area with the latest lines
            UpdateTextArea(hText);

            // Clear the edit control
            SetWindowText(hwnd, TEXT(""));

            return 0; // Prevent further processing of the ENTER key
        }
    }
    // Call the original window procedure for default processing
    return DefSubclassProc(hwnd, uMsg, wParam, lParam);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hEdit;
    static HWND hText;

    switch (uMsg) {
    case WM_CREATE: {
        // Create an edit control
        hEdit = CreateWindowEx(0, TEXT("EDIT"), TEXT(""),
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
            10, 10, 280, 20,
            hwnd, (HMENU)IDC_MAIN_EDIT, GetModuleHandle(NULL), NULL);

        // Create a text area (static control)
        hText = CreateWindowEx(0, TEXT("STATIC"), TEXT(""),
            WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOTIFY,
            10, 40, 300, 400,
            hwnd, (HMENU)IDC_MAIN_TEXT, GetModuleHandle(NULL), NULL);

        // Set text area background to black and text to white
        SetBkColor(GetDC(hText), RGB(0, 0, 0));
        SetTextColor(GetDC(hText), RGB(255, 255, 255));

        // Subclass the edit control to handle ENTER key
        SetWindowSubclass(hEdit, EditProc, 0, (DWORD_PTR)hText);

        break;
    }

    case WM_DESTROY: {
        PostQuitMessage(0);
        break;
    }

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    const TCHAR CLASS_NAME[] = TEXT("Simple Window Class");

    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, TEXT("Simple Window"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 330, 500,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}




/*
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <tchar.h>
#include <string.h>
#include <commctrl.h>

#pragma comment(lib, "Comctl32.lib")

#define IDC_MAIN_EDIT 101   // Edit control identifier
#define IDC_MAIN_TEXT 102   // Text area identifier

// Buffer to store all lines
TCHAR allTextBuffer[4096] = TEXT(""); // Initial buffer size

LRESULT CALLBACK EditProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam, UINT_PTR uIdSubclass, DWORD_PTR dwRefData) {
    if (uMsg == WM_KEYDOWN) {
        if (wParam == VK_RETURN) {
            HWND hText = (HWND)dwRefData;
            TCHAR textBuffer[1024];

            // Get the text from the edit control
            GetWindowText(hwnd, textBuffer, 1024);

            // Add a new line to the accumulated text
            _tcscat(allTextBuffer, textBuffer);
            _tcscat(allTextBuffer, TEXT("\r\n")); // Add a newline after each entry

            // Set the accumulated text to the text area (static control)
            SetWindowText(hText, allTextBuffer);

            // Clear the edit control
            SetWindowText(hwnd, TEXT(""));

            return 0; // Prevent further processing of the ENTER key
        }
    }
    // Call the original window procedure for default processing
    return DefSubclassProc(hwnd, uMsg, wParam, lParam);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hEdit;
    static HWND hText;

    switch (uMsg) {
    case WM_CREATE: {
        // Create an edit control
        hEdit = CreateWindowEx(0, TEXT("EDIT"), TEXT(""),
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
            10, 10, 280, 20,
            hwnd, (HMENU)IDC_MAIN_EDIT, GetModuleHandle(NULL), NULL);

        // Create a text area (static control)
        hText = CreateWindowEx(0, TEXT("STATIC"), TEXT(""),
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            10, 40, 300, 400,
            hwnd, (HMENU)IDC_MAIN_TEXT, GetModuleHandle(NULL), NULL);

        // Set text area background to black and text to white
        SetBkColor(GetDC(hText), RGB(0, 0, 0));
        SetTextColor(GetDC(hText), RGB(255, 255, 255));

        // Subclass the edit control to handle ENTER key
        SetWindowSubclass(hEdit, EditProc, 0, (DWORD_PTR)hText);

        break;
    }

    case WM_DESTROY: {
        PostQuitMessage(0);
        break;
    }

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    const TCHAR CLASS_NAME[] = TEXT("Simple Window Class");

    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, TEXT("Simple Window"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 330, 500,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
*/




/*
#include <windows.h>
#include <tchar.h>
#include <commctrl.h>

#pragma comment(lib, "Comctl32.lib")

#define IDC_MAIN_EDIT 101   // Edit control identifier
#define IDC_MAIN_TEXT 102   // Text area identifier

LRESULT CALLBACK EditProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam, UINT_PTR uIdSubclass, DWORD_PTR dwRefData) {
    if (uMsg == WM_KEYDOWN && wParam == VK_RETURN) {
        HWND hText = (HWND)dwRefData;
        TCHAR textBuffer[1024];

        // Get the text from the edit control
        GetWindowText(hwnd, textBuffer, 1024);

        // Set the text to the text area (static control)
        SetWindowText(hText, textBuffer);

        // Clear the edit control
        SetWindowText(hwnd, TEXT(""));

        return 0; // Prevent further processing of the ENTER key
    }

    return DefSubclassProc(hwnd, uMsg, wParam, lParam);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static HWND hEdit;
    static HWND hText;

    switch (uMsg) {
    case WM_CREATE: {
        // Create an edit control
        hEdit = CreateWindowEx(0, TEXT("EDIT"), TEXT(""),
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
            10, 10, 280, 20,
            hwnd, (HMENU)IDC_MAIN_EDIT, GetModuleHandle(NULL), NULL);

        // Create a text area (static control)
        hText = CreateWindowEx(0, TEXT("STATIC"), TEXT(""),
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            10, 40, 300, 400,
            hwnd, (HMENU)IDC_MAIN_TEXT, GetModuleHandle(NULL), NULL);

        // Set text area background to black and text to white
        SetBkColor(GetDC(hText), RGB(0, 0, 0));
        SetTextColor(GetDC(hText), RGB(255, 255, 255));

        // Subclass the edit control to handle ENTER key press
        SetWindowSubclass(hEdit, EditProc, 0, (DWORD_PTR)hText);

        break;
    }

    case WM_DESTROY: {
        RemoveWindowSubclass(hEdit, EditProc, 0);
        PostQuitMessage(0);
        break;
    }

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    const TCHAR CLASS_NAME[] = TEXT("Simple Window Class");

    WNDCLASS wc = { 0 };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, CLASS_NAME, TEXT("Simple Window"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 330, 500,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
*/





#if 0
// win32_cmd_version.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "win32_cmd_version.h"

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
    LoadStringW(hInstance, IDC_WIN32CMDVERSION, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_WIN32CMDVERSION));

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
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_WIN32CMDVERSION));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_WIN32CMDVERSION);
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