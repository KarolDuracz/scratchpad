#include <windows.h>
#include <string>

#include "framework.h"
#include "calc_exe_win32_example_chatgpt.h"

const char g_szClassName[] = "MyCalculatorClass";
HWND hDisplay;  // Pole tekstowe do wyœwietlania wyników

// Funkcja dodaj¹ca przyciski
HWND AddButton(HWND hwnd, const char* text, int x, int y, int width, int height, int id)
{
    return CreateWindow(
        L"BUTTON", (LPCWSTR)text, WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        x, y, width, height, hwnd, (HMENU)id, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);
}

// Funkcja dodaj¹ca pole tekstowe (wyœwietlacz)
HWND AddDisplay(HWND hwnd, int x, int y, int width, int height)
{
    return CreateWindow(
        L"EDIT", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_RIGHT | ES_READONLY,
        x, y, width, height, hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);
}

// Obs³uga przycisków
void OnButtonClick(HWND hwnd, int id)
{
    char buffer[256];
    GetWindowText(hDisplay, (LPWSTR)buffer, 256);

    std::string text = buffer;
    if (id >= 0 && id <= 9)
    {
        text += std::to_string(id);
    }
    else
    {
        switch (id)
        {
        case 100:  // Przycisk C (Clear)
            text = "";
            break;
        case 101:  // Przycisk +
            text += " + ";
            break;
        case 102:  // Przycisk -
            text += " - ";
            break;
        case 103:  // Przycisk *
            text += " * ";
            break;
        case 104:  // Przycisk /
            text += " / ";
            break;
        case 105:  // Przycisk =
            try
            {
                double result = std::stod(text);  // Prosta wersja obliczeñ
                std::wstring wstr = std::to_wstring(result);
                //SetWindowText(hDisplay, std::to_string(result).c_str());
                SetWindowText(hDisplay, wstr.c_str());
                return;
            }
            catch (...)
            {
                text = "Error";
            }
            break;
        }
    }
    //std::wstring ws2 = std::to_wstring(text);
    //SetWindowText(hDisplay, ws2.c_str());
}

// Procedura obs³ugi okna
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CREATE:
        // Tworzenie przycisków i wyœwietlacza
        hDisplay = AddDisplay(hwnd, 10, 10, 230, 30);

        AddButton(hwnd, "7", 10, 50, 50, 50, 7);
        AddButton(hwnd, "8", 70, 50, 50, 50, 8);
        AddButton(hwnd, "9", 130, 50, 50, 50, 9);
        AddButton(hwnd, "/", 190, 50, 50, 50, 104);

        AddButton(hwnd, "4", 10, 110, 50, 50, 4);
        AddButton(hwnd, "5", 70, 110, 50, 50, 5);
        AddButton(hwnd, "6", 130, 110, 50, 50, 6);
        AddButton(hwnd, "*", 190, 110, 50, 50, 103);

        AddButton(hwnd, "1", 10, 170, 50, 50, 1);
        AddButton(hwnd, "2", 70, 170, 50, 50, 2);
        AddButton(hwnd, "3", 130, 170, 50, 50, 3);
        AddButton(hwnd, "-", 190, 170, 50, 50, 102);

        AddButton(hwnd, "0", 10, 230, 110, 50, 0);
        AddButton(hwnd, "C", 130, 230, 50, 50, 100);
        AddButton(hwnd, "+", 190, 230, 50, 50, 101);

        AddButton(hwnd, "=", 10, 290, 230, 50, 105);
        break;

    case WM_COMMAND:
        OnButtonClick(hwnd, LOWORD(wParam));
        break;

    case WM_CLOSE:
        DestroyWindow(hwnd);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

// Punkt wejœcia
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    WNDCLASSEX wc;
    HWND hwnd;
    MSG Msg;

    // Rejestracja klasy okna
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = 0;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = (LPCWSTR)g_szClassName;
    wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassEx(&wc))
    {
        MessageBox(NULL, L"Window Registration Failed!", L"Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // Tworzenie okna
    hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        (LPCWSTR)g_szClassName,
        L"Simple Scientific Calculator",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 260, 400,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL)
    {
        MessageBox(NULL, L"Window Creation Failed!", L"Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // Pêtla wiadomoœci
    while (GetMessage(&Msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    return Msg.wParam;
}


#if 0
// calc_exe_win32_example_chatgpt.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "calc_exe_win32_example_chatgpt.h"

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
    LoadStringW(hInstance, IDC_CALCEXEWIN32EXAMPLECHATGPT, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_CALCEXEWIN32EXAMPLECHATGPT));

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
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_CALCEXEWIN32EXAMPLECHATGPT));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_CALCEXEWIN32EXAMPLECHATGPT);
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
