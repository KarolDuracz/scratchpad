
#include "framework.h"
#include "remote_control_menu_tv.h"

#include <windows.h>
#include <tchar.h>
#include <wingdi.h>

#pragma comment(lib, "Msimg32.lib")

// Menu items
LPCWSTR menuItems[] = {
    L"Program 1",
    L"Program 2",
    L"Program 3",
    L"Program 4",
    L"Program 5"
};
const int menuItemCount = sizeof(menuItems) / sizeof(menuItems[0]);
int selectedItem = 0; // Index of selected menu item

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE:
        // Set up the initial state if needed
        break;

    case WM_KEYDOWN: {
        switch (wParam) {
        case VK_UP:
            selectedItem = (selectedItem - 1 + menuItemCount) % menuItemCount;
            InvalidateRect(hwnd, NULL, TRUE); // Request a redraw
            break;

        case VK_DOWN:
            selectedItem = (selectedItem + 1) % menuItemCount;
            InvalidateRect(hwnd, NULL, TRUE); // Request a redraw
            break;

        case VK_RETURN:
            MessageBox(hwnd, menuItems[selectedItem], L"Program Selected", MB_OK);
            break;
        }
        break;
    }

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        // Simulate a transparent TV screen effect with a gradient background
        RECT clientRect;
        GetClientRect(hwnd, &clientRect);
        TRIVERTEX vertex[2];
        GRADIENT_RECT gRect = { 0, 1 };

        vertex[0].x = 0;
        vertex[0].y = 0;
        vertex[0].Red = 0x0000;
        vertex[0].Green = 0x0000;
        vertex[0].Blue = 0x8000;
        vertex[0].Alpha = 0x4000;

        vertex[1].x = clientRect.right;
        vertex[1].y = clientRect.bottom;
        vertex[1].Red = 0x0000;
        vertex[1].Green = 0x0000;
        vertex[1].Blue = 0x0000;
        vertex[1].Alpha = 0x4000;

        GradientFill(hdc, vertex, 2, &gRect, 1, GRADIENT_FILL_RECT_V);

        // Draw the menu items
        for (int i = 0; i < menuItemCount; i++) {
            RECT rect = { 50, 50 + i * 50, clientRect.right - 50, 90 + i * 50 };

            if (i == selectedItem) {
                // Highlight the selected item
                HBRUSH hBrush = CreateSolidBrush(RGB(0, 120, 215));
                FillRect(hdc, &rect, hBrush);
                DeleteObject(hBrush);
                SetTextColor(hdc, RGB(255, 255, 255));
            }
            else {
                SetTextColor(hdc, RGB(200, 200, 200));
            }

            SetBkMode(hdc, TRANSPARENT);
            DrawText(hdc, menuItems[i], -1, &rect, DT_SINGLELINE | DT_VCENTER | DT_CENTER);
        }

        EndPaint(hwnd, &ps);
        break;
    }

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0, 0, hInstance, NULL, LoadCursor(NULL, IDC_ARROW), NULL, NULL, _T("MenuExample"), NULL };
    RegisterClassEx(&wc);

    HWND hwnd = CreateWindowEx(WS_EX_LAYERED, _T("MenuExample"), _T("TV Program Selector"), WS_OVERLAPPEDWINDOW, 100, 100, 400, 400, NULL, NULL, wc.hInstance, NULL);
    SetLayeredWindowAttributes(hwnd, 0, (255 * 90) / 100, LWA_ALPHA); // Set transparency to 90%
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return msg.wParam;
}



#if 0
// remote_control_menu_tv.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "remote_control_menu_tv.h"

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
    LoadStringW(hInstance, IDC_REMOTECONTROLMENUTV, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_REMOTECONTROLMENUTV));

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
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_REMOTECONTROLMENUTV));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_REMOTECONTROLMENUTV);
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