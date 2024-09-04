#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <commctrl.h>
#include <stdio.h>
#include <string.h>

#pragma comment(lib,  "Comctl32.lib")

#define IDM_NEW_FILE 1
#define IDM_NEW_FOLDER 2
#define IDM_DELETE 3
#define IDM_COPY 4
#define IDM_MOVE 5
#define IDM_PROPERTIES 6

#define MAX_TEXT_LENGTH 1024

wchar_t textBuffer[MAX_TEXT_LENGTH] = L"";

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void DrawFileSystem(HWND hwnd, HDC hdc);
void CreateNewItem(const wchar_t* name, int type, int x, int y);
void ShowContextMenu(HWND hwnd, POINT pt);
void ShowProperties(HWND hwnd, int selectedItem);
void DrawTextArea(HWND hwnd, HDC hdc);

typedef struct {
    wchar_t name[256];
    int type; // 0: Folder, 1: File
    RECT rect; // Position and size of icon
} FileSystemItem;

#define MAX_ITEMS 100
FileSystemItem items[MAX_ITEMS];
int itemCount = 0;

HINSTANCE hInst;
HWND hEdit;
wchar_t editText[256] = L"";

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
    const wchar_t CLASS_NAME[] = L"FileSystemWindowClass";

    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        L"Simple File System",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    // Initialize common controls
    INITCOMMONCONTROLSEX icex;
    icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
    icex.dwICC = ICC_WIN95_CLASSES;
    InitCommonControlsEx(&icex);

    // Create an Edit control
    hEdit = CreateWindowEx(
        0, L"EDIT", L"",
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT,
        50, 400, 200, 25,
        hwnd, NULL, hInstance, NULL
    );

    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static POINT clickPoint;
    static POINT dragPoint;
    static int selectedItem = -1;
    static BOOL dragging = FALSE;

    switch (uMsg) {
    case WM_CREATE:
        // Create root folder
        CreateNewItem(L"Root", 0, 50, 50);
        break;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        // without double buffering this is the simple way to clear rect with noise 
        FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));

        // Draw file system items
        DrawFileSystem(hwnd, hdc);

        // Draw the text area
        DrawTextArea(hwnd, hdc);

        EndPaint(hwnd, &ps);
    } break;

    case WM_LBUTTONDOWN: {
        clickPoint.x = LOWORD(lParam);
        clickPoint.y = HIWORD(lParam);

        // Detect item selection
        for (int i = 0; i < itemCount; i++) {
            if (PtInRect(&items[i].rect, clickPoint)) {
                selectedItem = i;
                dragPoint = clickPoint;
                dragging = TRUE;
                SetCapture(hwnd);
                break;
            }
        }

        InvalidateRect(hwnd, NULL, TRUE);
    } break;

    case WM_MOUSEMOVE: {
        if (dragging && selectedItem >= 0) {
            POINT currentPoint = { LOWORD(lParam), HIWORD(lParam) };
            int dx = currentPoint.x - dragPoint.x;
            int dy = currentPoint.y - dragPoint.y;

            OffsetRect(&items[selectedItem].rect, dx, dy);
            dragPoint = currentPoint;

            InvalidateRect(hwnd, NULL, TRUE);
        }
    } break;

    case WM_LBUTTONUP: {
        if (dragging) {
            dragging = FALSE;
            ReleaseCapture();
            InvalidateRect(hwnd, NULL, TRUE);
        }
    } break;

    case WM_RBUTTONDOWN: {
        POINT pt;
        pt.x = LOWORD(lParam);
        pt.y = HIWORD(lParam);
        ShowContextMenu(hwnd, pt);
    } break;

    case WM_COMMAND: {
        switch (LOWORD(wParam)) {
        case IDM_NEW_FILE:
            CreateNewItem(L"New File", 1, clickPoint.x + 10, clickPoint.y + 10);
            InvalidateRect(hwnd, NULL, TRUE);
            break;
        case IDM_NEW_FOLDER:
            CreateNewItem(L"New Folder", 0, clickPoint.x + 10, clickPoint.y + 10);
            InvalidateRect(hwnd, NULL, TRUE);
            break;
        case IDM_DELETE:
            if (selectedItem >= 0 && selectedItem < itemCount) {
                for (int i = selectedItem; i < itemCount - 1; i++) {
                    items[i] = items[i + 1];
                }
                itemCount--;
                selectedItem = -1;
                InvalidateRect(hwnd, NULL, TRUE);
            }
            break;
        case IDM_PROPERTIES:
            if (selectedItem >= 0 && selectedItem < itemCount) {
                ShowProperties(hwnd, selectedItem);
            }
            break;
        }
    } break;

    case WM_KEYDOWN: {
        if (wParam == VK_RETURN) {
            // Get text from edit control
            GetWindowText(hEdit, editText, sizeof(editText) / sizeof(editText[0]));

            // Append the text to the text buffer
            if (wcslen(textBuffer) > 0) {
                wcscat(textBuffer, L"\n"); // Add a newline if buffer is not empty
            }
            wcscat(textBuffer, editText);

            // Clear the edit control
            SetWindowText(hEdit, L"");

            // Invalidate the window to trigger a repaint
            InvalidateRect(hwnd, NULL, TRUE);
        }
    } break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

void DrawFileSystem(HWND hwnd, HDC hdc) {
    for (int i = 0; i < itemCount; i++) {
        RECT r = items[i].rect;
        FillRect(hdc, &r, (items[i].type == 0) ? GetSysColorBrush(COLOR_WINDOWTEXT) : GetSysColorBrush(COLOR_ACTIVECAPTION));

        SetTextColor(hdc, (items[i].type == 0) ? RGB(255, 255, 255) : RGB(0, 0, 0));
        DrawText(hdc, items[i].name, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }
}

void CreateNewItem(const wchar_t* name, int type, int x, int y) {
    if (itemCount >= MAX_ITEMS) return;

    FileSystemItem* item = &items[itemCount++];
    wcscpy(item->name, name);
    item->type = type;
    item->rect.left = x;
    item->rect.top = y;
    item->rect.right = x + 100;
    item->rect.bottom = y + 50;
}

void ShowContextMenu(HWND hwnd, POINT pt) {
    HMENU hMenu = CreatePopupMenu();
    AppendMenu(hMenu, MF_STRING, IDM_NEW_FILE, L"New File");
    AppendMenu(hMenu, MF_STRING, IDM_NEW_FOLDER, L"New Folder");
    AppendMenu(hMenu, MF_STRING, IDM_DELETE, L"Delete");
    AppendMenu(hMenu, MF_SEPARATOR, 0, NULL);
    AppendMenu(hMenu, MF_STRING, IDM_PROPERTIES, L"Properties");

    // Display the context menu at the cursor's position
    ClientToScreen(hwnd, &pt);
    TrackPopupMenu(hMenu, TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, NULL);
    DestroyMenu(hMenu);
}

void ShowProperties(HWND hwnd, int selectedItem) {
    wchar_t buffer[512];
    FileSystemItem* item = &items[selectedItem];
    swprintf(buffer, 512, L"Name: %s\nType: %s\nPosition: (%d, %d)",
        item->name,
        (item->type == 0) ? L"Folder" : L"File",
        item->rect.left, item->rect.top);

    MessageBox(hwnd, buffer, L"Properties", MB_OK | MB_ICONINFORMATION);
}

void DrawTextArea(HWND hwnd, HDC hdc) {
    // Define the text area (right side, 400x300 px, black background)
    RECT textArea = { 400, 0, 800, 300 };

    // Fill the text area with black background
    HBRUSH hBrush = CreateSolidBrush(RGB(125, 125, 0));
    FillRect(hdc, &textArea, hBrush);
    DeleteObject(hBrush);

    // Set text color to white
    SetTextColor(hdc, RGB(255, 255, 255));

    // Set font
    HFONT hFont = CreateFont(
        20,                    // Height of font
        0,                     // Width of font (0 = default width)
        0,                     // Angle of escapement
        0,                     // Base line orientation angle
        FW_NORMAL,             // Font weight
        FALSE,                 // Italic
        FALSE,                 // Underline
        FALSE,                 // Strikeout
        DEFAULT_CHARSET,       // Character set identifier
        OUT_OUTLINE_PRECIS,    // Output precision
        CLIP_DEFAULT_PRECIS,   // Clipping precision
        CLEARTYPE_QUALITY,     // Output quality
        VARIABLE_PITCH,        // Pitch and family
        L"Arial"               // Font name
    );

    HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);

    // Draw the text from the text buffer in the text area
    RECT textRect = textArea;
    //DrawText(hdc, textBuffer, -1, &textRect, DT_LEFT | DT_TOP | DT_WORDBREAK | DT_CALCRECT);

    SetTextColor(hdc, RGB(255, 155, 255));
    DrawText(hdc, L"test", -1, &textRect, DT_LEFT | DT_TOP | DT_WORDBREAK);

    // Clean up
    SelectObject(hdc, hOldFont);
    DeleteObject(hFont);
}




/*
        
        #define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <commctrl.h>
#include <stdio.h>
#include <string.h>

#pragma comment(lib,  "Comctl32.lib")

#define IDM_NEW_FILE 1
#define IDM_NEW_FOLDER 2
#define IDM_DELETE 3
#define IDM_COPY 4
#define IDM_MOVE 5
#define IDM_PROPERTIES 6

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void DrawFileSystem(HWND hwnd, HDC hdc);
void CreateNewItem(const wchar_t* name, int type, int x, int y);
void ShowContextMenu(HWND hwnd, POINT pt);
void ShowProperties(HWND hwnd, int selectedItem);

typedef struct {
    wchar_t name[256];
    int type; // 0: Folder, 1: File
    RECT rect; // Position and size of icon
} FileSystemItem;

#define MAX_ITEMS 100
FileSystemItem items[MAX_ITEMS];
int itemCount = 0;

HINSTANCE hInst;

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
    const wchar_t CLASS_NAME[] = L"FileSystemWindowClass";

    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        L"Simple File System",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    // Initialize common controls
    INITCOMMONCONTROLSEX icex;
    icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
    icex.dwICC = ICC_WIN95_CLASSES;
    InitCommonControlsEx(&icex);

    MSG msg = { 0 };
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static POINT clickPoint;
    static POINT dragPoint;
    static int selectedItem = -1;
    static BOOL dragging = FALSE;

    switch (uMsg) {

   

    case WM_CREATE:
        // Create root folder
        CreateNewItem(L"Root", 0, 50, 50);
        break;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        // without double buffering this is the simple way to clear rect with noise 
        FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));

        DrawFileSystem(hwnd, hdc);

        EndPaint(hwnd, &ps);
    } break;

    case WM_LBUTTONDOWN: {
        clickPoint.x = LOWORD(lParam);
        clickPoint.y = HIWORD(lParam);

        // Detect item selection
        for (int i = 0; i < itemCount; i++) {
            if (PtInRect(&items[i].rect, clickPoint)) {
                selectedItem = i;
                dragPoint = clickPoint;
                dragging = TRUE;
                SetCapture(hwnd);
                break;
            }
        }

        InvalidateRect(hwnd, NULL, TRUE);
    } break;

    case WM_MOUSEMOVE: {
        if (dragging && selectedItem >= 0) {
            POINT currentPoint = { LOWORD(lParam), HIWORD(lParam) };
            int dx = currentPoint.x - dragPoint.x;
            int dy = currentPoint.y - dragPoint.y;

            OffsetRect(&items[selectedItem].rect, dx, dy);
            dragPoint = currentPoint;

            InvalidateRect(hwnd, NULL, TRUE);
        }
    } break;

    case WM_LBUTTONUP: {
        if (dragging) {
            dragging = FALSE;
            ReleaseCapture();
            InvalidateRect(hwnd, NULL, TRUE);
        }
    } break;

    case WM_RBUTTONDOWN: {
        POINT pt;
        pt.x = LOWORD(lParam);
        pt.y = HIWORD(lParam);
        ShowContextMenu(hwnd, pt);
    } break;

    case WM_COMMAND: {
        switch (LOWORD(wParam)) {
        case IDM_NEW_FILE:
            CreateNewItem(L"New File", 1, clickPoint.x + 10, clickPoint.y + 10);
            InvalidateRect(hwnd, NULL, TRUE);
            break;
        case IDM_NEW_FOLDER:
            CreateNewItem(L"New Folder", 0, clickPoint.x + 10, clickPoint.y + 10);
            InvalidateRect(hwnd, NULL, TRUE);
            break;
        case IDM_DELETE:
            if (selectedItem >= 0 && selectedItem < itemCount) {
                for (int i = selectedItem; i < itemCount - 1; i++) {
                    items[i] = items[i + 1];
                }
                itemCount--;
                selectedItem = -1;
                InvalidateRect(hwnd, NULL, TRUE);
            }
            break;
        case IDM_PROPERTIES:
            if (selectedItem >= 0 && selectedItem < itemCount) {
                ShowProperties(hwnd, selectedItem);
            }
            break;
        }
    } break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

void DrawFileSystem(HWND hwnd, HDC hdc) {
    for (int i = 0; i < itemCount; i++) {
        RECT r = items[i].rect;
        FillRect(hdc, &r, (items[i].type == 0) ? GetSysColorBrush(COLOR_WINDOWTEXT) : GetSysColorBrush(COLOR_ACTIVECAPTION));

        SetTextColor(hdc, (items[i].type == 0) ? RGB(255, 255, 255) : RGB(0, 0, 0));
        DrawText(hdc, items[i].name, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }
}

void CreateNewItem(const wchar_t* name, int type, int x, int y) {
    if (itemCount >= MAX_ITEMS) return;

    FileSystemItem* item = &items[itemCount++];
    wcscpy(item->name, name);
    item->type = type;
    item->rect.left = x;
    item->rect.top = y;
    item->rect.right = x + 100;
    item->rect.bottom = y + 50;
}

void ShowContextMenu(HWND hwnd, POINT pt) {
    HMENU hMenu = CreatePopupMenu();
    AppendMenu(hMenu, MF_STRING, IDM_NEW_FILE, L"New File");
    AppendMenu(hMenu, MF_STRING, IDM_NEW_FOLDER, L"New Folder");
    AppendMenu(hMenu, MF_STRING, IDM_DELETE, L"Delete");
    AppendMenu(hMenu, MF_SEPARATOR, 0, NULL);
    AppendMenu(hMenu, MF_STRING, IDM_PROPERTIES, L"Properties");

    // Display the context menu at the cursor's position
    ClientToScreen(hwnd, &pt);
    TrackPopupMenu(hMenu, TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, NULL);
    DestroyMenu(hMenu);
}

void ShowProperties(HWND hwnd, int selectedItem) {
    wchar_t buffer[512];
    FileSystemItem* item = &items[selectedItem];
    swprintf(buffer, 512, L"Name: %s\nType: %s\nPosition: (%d, %d)",
        item->name,
        (item->type == 0) ? L"Folder" : L"File",
        item->rect.left, item->rect.top);

    MessageBox(hwnd, buffer, L"Properties", MB_OK | MB_ICONINFORMATION);
}





#if 0
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <commctrl.h>
#include <stdio.h>
#include <string.h>

#pragma comment(lib, "Comctl32.lib")

//#include "framework.h"
//#include "file_system_win32_demo.h"

#define IDM_NEW_FILE 1
#define IDM_NEW_FOLDER 2
#define IDM_DELETE 3
#define IDM_COPY 4
#define IDM_MOVE 5

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void DrawFileSystem(HWND hwnd, HDC hdc);
void CreateNewItem(const char* name, int type, int x, int y);
void ShowContextMenu(HWND hwnd, POINT pt);

typedef struct {
    char name[256];
    int type; // 0: Folder, 1: File
    RECT rect; // Position and size of icon
} FileSystemItem;

#define MAX_ITEMS 100
FileSystemItem items[MAX_ITEMS];
int itemCount = 0;

HINSTANCE hInst;

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    const char CLASS_NAME[] = "FileSystemWindowClass";

    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCWSTR)CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0,
        (LPCWSTR)CLASS_NAME,
        L"Simple File System",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        NULL,
        NULL,
        hInstance,
        NULL
    );

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

// LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static POINT clickPoint;
    static int selectedItem = -1;

    switch (uMsg) {
    case WM_CREATE:
        InitCommonControls();

        // Create root folder
        CreateNewItem("Root", 0, 50, 50);
        break;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        DrawFileSystem(hwnd, hdc);

        EndPaint(hwnd, &ps);
    } break;

    case WM_LBUTTONDOWN: {
        clickPoint.x = LOWORD(lParam);
        clickPoint.y = HIWORD(lParam);

        // Detect item selection
        for (int i = 0; i < itemCount; i++) {
            if (PtInRect(&items[i].rect, clickPoint)) {
                selectedItem = i;
                break;
            }
        }

        InvalidateRect(hwnd, NULL, TRUE);
    } break;

    case WM_RBUTTONDOWN: {
        POINT pt;
        pt.x = LOWORD(lParam);
        pt.y = HIWORD(lParam);
        ShowContextMenu(hwnd, pt);
    } break;

    case WM_COMMAND: {
        switch (LOWORD(wParam)) {
        case IDM_NEW_FILE:
            CreateNewItem("New File", 1, clickPoint.x + 10, clickPoint.y + 10);
            InvalidateRect(hwnd, NULL, TRUE);
            break;
        case IDM_NEW_FOLDER:
            CreateNewItem("New Folder", 0, clickPoint.x + 10, clickPoint.y + 10);
            InvalidateRect(hwnd, NULL, TRUE);
            break;
        case IDM_DELETE:
            if (selectedItem >= 0 && selectedItem < itemCount) {
                for (int i = selectedItem; i < itemCount - 1; i++) {
                    items[i] = items[i + 1];
                }
                itemCount--;
                selectedItem = -1;
                InvalidateRect(hwnd, NULL, TRUE);
            }
            break;
        }
    } break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

void DrawFileSystem(HWND hwnd, HDC hdc) {
    for (int i = 0; i < itemCount; i++) {
        RECT r = items[i].rect;
        FillRect(hdc, &r, (items[i].type == 0) ? GetSysColorBrush(COLOR_WINDOWTEXT) : GetSysColorBrush(COLOR_ACTIVECAPTION));

        SetTextColor(hdc, (items[i].type == 0) ? RGB(255, 255, 255) : RGB(0, 0, 0));
        DrawText(hdc, (LPCWSTR)items[i].name, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }
}

void CreateNewItem(const char* name, int type, int x, int y) {
    if (itemCount >= MAX_ITEMS) return;

    FileSystemItem* item = &items[itemCount++];
    strcpy(item->name, name);
    item->type = type;
    item->rect.left = x;
    item->rect.top = y;
    item->rect.right = x + 100;
    item->rect.bottom = y + 50;
}

void ShowContextMenu(HWND hwnd, POINT pt) {
    HMENU hMenu = CreatePopupMenu();
    AppendMenu(hMenu, MF_STRING, IDM_NEW_FILE, L"New File");
    AppendMenu(hMenu, MF_STRING, IDM_NEW_FOLDER, L"New Folder");
    AppendMenu(hMenu, MF_STRING, IDM_DELETE, L"Delete");

    TrackPopupMenu(hMenu, TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, NULL);
    DestroyMenu(hMenu);
}
#endif


#if 0
// file_system_win32_demo.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "file_system_win32_demo.h"



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
    LoadStringW(hInstance, IDC_FILESYSTEMWIN32DEMO, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_FILESYSTEMWIN32DEMO));

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
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_FILESYSTEMWIN32DEMO));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_FILESYSTEMWIN32DEMO);
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
        
*/