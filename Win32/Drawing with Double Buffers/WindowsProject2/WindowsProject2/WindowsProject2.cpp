#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <time.h>
#include <stdlib.h>  // Do funkcji rand()
#include "framework.h"
#include "WindowsProject2.h"
#include <cstdlib>
#include <intsafe.h>

#include <windows.h>
#include <time.h>
#include <stdlib.h>  // Do funkcji rand()

const char g_szClassName[] = "MyWindowClass";

// Rozmiar okna
#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 400

#define GRAPH_MAX_Y_VALUE 100  // Maksymalna wartoœæ na osi Y
#define SECONDS_IN_DAY 86400   // Liczba sekund w ci¹gu doby

int graphData[WINDOW_WIDTH] = { 0 };  // Tablica przechowuj¹ca wartoœci wykresu

int cnt = 0;

// Funkcja rysuj¹ca wykres z podwójnym buforowaniem
void DrawGraph(HDC hdc, RECT rect, int second)
{
    // Wysokoœæ wykresu
    int height = rect.bottom - rect.top;
    int width = rect.right - rect.left;

    // Tworzenie pamiêciowego bufora do rysowania (podwójne buforowanie)
    HDC memDC = CreateCompatibleDC(hdc);
    HBITMAP memBitmap = CreateCompatibleBitmap(hdc, width, height);
    SelectObject(memDC, memBitmap);

    // Wype³nij t³o na bia³o
    FillRect(memDC, &rect, (HBRUSH)(COLOR_HIGHLIGHT + 1));

    // Ustawienia pióra (linia wykresu)
    HPEN hPen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
    SelectObject(memDC, hPen);

    // Rysowanie osi X i Y
    MoveToEx(memDC, rect.left, rect.bottom, NULL);
    LineTo(memDC, rect.right, rect.bottom);  // Oœ X

    MoveToEx(memDC, rect.left, rect.top, NULL);
    LineTo(memDC, rect.left, rect.bottom);  // Oœ Y

    // Dodanie podzia³ki pionowej
    for (int i = 0; i <= GRAPH_MAX_Y_VALUE; i += 20)
    {
        int y = rect.bottom - (i * height / GRAPH_MAX_Y_VALUE);
        MoveToEx(memDC, rect.left - 5, y, NULL);
        LineTo(memDC, rect.left, y);

        // Rysowanie wartoœci podzia³ki
        //char label[10];
        //wsprintf(label, "%d", i);
        //TextOut(memDC, rect.left - 30, y - 8, label, strlen(label));
    }

    // Przesuwanie danych w lewo, aby symulowaæ ruch wykresu
    for (int i = 0; i < width - 1; i++)
    {
        graphData[i] = graphData[i + 1];
    }

    // Generowanie nowej losowej wartoœci na osi Y
    if (rand() < 1000) {
        for (int i = 0; i < 20; i++) {
            graphData[width - (1 + i)] = (GRAPH_MAX_Y_VALUE / 100 + 1);
            TextOut(memDC, (width - cnt) - 100, 20, L"text", sizeof("text"));
        }
    }
    else {
        graphData[width - 1] = (GRAPH_MAX_Y_VALUE / 2 + 1);
    }   
    
    // inc cnt
    if (cnt > 100) {
        cnt = 0;
    }
    cnt++;

    // Rysowanie wykresu
    for (int i = 0; i < width - 1; i++)
    {
        int x1 = rect.left + i;
        int y1 = rect.bottom - (graphData[i] * height / GRAPH_MAX_Y_VALUE);

        int x2 = rect.left + i + 1;
        int y2 = rect.bottom - (graphData[i + 1] * height / GRAPH_MAX_Y_VALUE);

        MoveToEx(memDC, x1, y1, NULL);
        LineTo(memDC, x2, y2);
    }

    // Przeniesienie zawartoœci pamiêciowego bufora na ekran
    BitBlt(hdc, 0, 0, width, height, memDC, 0, 0, SRCCOPY);

    // Czyszczenie pamiêci
    DeleteObject(hPen);
    DeleteObject(memBitmap);
    DeleteDC(memDC);
}

// Procedura obs³ugi okna
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT rect;
        GetClientRect(hwnd, &rect);

        // Pobranie bie¿¹cego czasu w sekundach od pó³nocy
        time_t now = time(0);
        struct tm timeinfo;
        localtime_s(&timeinfo, &now);
        int seconds = timeinfo.tm_hour * 3600 + timeinfo.tm_min * 60 + timeinfo.tm_sec;

        // Rysowanie wykresu
        DrawGraph(hdc, rect, seconds);

        EndPaint(hwnd, &ps);
    }
    break;
    case WM_TIMER:
        InvalidateRect(hwnd, NULL, FALSE); // Odœwie¿anie okna
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
        L"Real-Time Graph with Vertical Scale and Double Buffering",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL)
    {
        MessageBox(NULL, L"Window Creation Failed!", L"Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // Ustawienie timera na odœwie¿anie co 1 ms
    SetTimer(hwnd, 1, 10, NULL);

    // Pêtla wiadomoœci
    while (GetMessage(&Msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    return Msg.wParam;
}



#if 0
// DOBRA WERSJA ALE WYKRES MIGOTA PRZY 1 ms albo poni¿ej 100ms nawet
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <time.h>
#include <stdlib.h>  // Do funkcji rand()
#include "framework.h"
#include "WindowsProject2.h"
#include <cstdlib>
#include <intsafe.h>

const char g_szClassName[] = "MyWindowClass";

// Rozmiar okna
#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 400

#define GRAPH_MAX_Y_VALUE 100  // Maksymalna wartoœæ na osi Y
#define SECONDS_IN_DAY 86400   // Liczba sekund w ci¹gu doby

int graphData[WINDOW_WIDTH] = { 0 };  // Tablica przechowuj¹ca wartoœci wykresu

// Funkcja rysuj¹ca wykres
void DrawGraph(HDC hdc, RECT rect, int second)
{
    // Wype³nij t³o na bia³o
    FillRect(hdc, &rect, (HBRUSH)(COLOR_WINDOW + 1));

    // Ustawienia pióra (linia wykresu)
    HPEN hPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255));
    SelectObject(hdc, hPen);

    // Wysokoœæ wykresu
    int height = rect.bottom - rect.top;
    int width = rect.right - rect.left;

    // Rysowanie osi X i Y
    MoveToEx(hdc, rect.left, rect.bottom, NULL);
    LineTo(hdc, rect.right, rect.bottom);  // Oœ X

    MoveToEx(hdc, rect.left, rect.top, NULL);
    LineTo(hdc, rect.left, rect.bottom);  // Oœ Y

    // Dodanie podzia³ki pionowej
    for (int i = 0; i <= GRAPH_MAX_Y_VALUE; i += 20)
    {
        int y = rect.bottom - (i * height / GRAPH_MAX_Y_VALUE);
        MoveToEx(hdc, rect.left - 5, y, NULL);
        LineTo(hdc, rect.left, y);

        // Rysowanie wartoœci podzia³ki
        //TCHAR label[10];
        char out[10];
        IntToChar(i, out);
        wchar_t label[10];
        std::mbstowcs(label, out, ARRAYSIZE(out));
        //TCHAR b[] = L"---sssss";
        //TCHAR label[] = L"---ssss";
        TextOut(hdc, rect.left - 30, y - 8, label, ARRAYSIZE(label));
    }

    // Przesuwanie danych w lewo, aby symulowaæ ruch wykresu
    for (int i = 0; i < width - 1; i++)
    {
        graphData[i] = graphData[i + 1];
    }

    // Generowanie nowej losowej wartoœci na osi Y
    //graphData[width - 1] = graphData[width] + (rand() % (GRAPH_MAX_Y_VALUE/2 + 1));
    //graphData[width - 1] = rand() % (GRAPH_MAX_Y_VALUE / 2 + 1);
    graphData[width - 1] = (GRAPH_MAX_Y_VALUE / 2 + 1) + log(rand());

    // Rysowanie wykresu
    for (int i = 0; i < width - 1; i++)
    {
        int x1 = rect.left + i;
        int y1 = rect.bottom - (graphData[i] * height / GRAPH_MAX_Y_VALUE);

        int x2 = rect.left + i + 1;
        int y2 = rect.bottom - (graphData[i + 1] * height / GRAPH_MAX_Y_VALUE);

        MoveToEx(hdc, x1, y1, NULL);
        LineTo(hdc, x2, y2);
    }

    DeleteObject(hPen);
}

// Procedura obs³ugi okna
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT rect;
        GetClientRect(hwnd, &rect);

        // Pobranie bie¿¹cego czasu w sekundach od pó³nocy
        time_t now = time(0);
        struct tm timeinfo;
        localtime_s(&timeinfo, &now);
        int seconds = timeinfo.tm_hour * 3600 + timeinfo.tm_min * 60 + timeinfo.tm_sec;

        // Rysowanie wykresu
        DrawGraph(hdc, rect, seconds);

        EndPaint(hwnd, &ps);
    }
    break;
    case WM_TIMER:
        InvalidateRect(hwnd, NULL, TRUE);
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
        L"Real-Time Graph with Vertical Scale",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL)
    {
        MessageBox(NULL, L"Window Creation Failed!", L"Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // Ustawienie timera na odœwie¿anie co sekundê
    SetTimer(hwnd, 1, 1, NULL);

    // Pêtla wiadomoœci
    while (GetMessage(&Msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    return Msg.wParam;
}
#endif


#if 0
#include <windows.h>
#include <time.h>
#include <stdlib.h>  // Do funkcji rand()
#include "framework.h"
#include "WindowsProject2.h"

const char g_szClassName[] = "MyWindowClass";

// Rozmiar okna
#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 400

#define GRAPH_MAX_Y_VALUE 100  // Maksymalna wartoœæ na osi Y
#define SECONDS_IN_DAY 86400   // Liczba sekund w ci¹gu doby

int graphData[WINDOW_WIDTH] = { 0 };  // Tablica przechowuj¹ca wartoœci wykresu

// Funkcja rysuj¹ca wykres
void DrawGraph(HDC hdc, RECT rect, int second)
{
    // Wype³nij t³o na bia³o
    FillRect(hdc, &rect, (HBRUSH)(COLOR_WINDOW + 1));

    // Ustawienia pióra (linia wykresu)
    HPEN hPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255));
    SelectObject(hdc, hPen);

    // Wysokoœæ wykresu
    int height = rect.bottom - rect.top;
    int width = rect.right - rect.left;

    // Rysowanie osi X i Y
    MoveToEx(hdc, rect.left, rect.bottom, NULL);
    LineTo(hdc, rect.right, rect.bottom);  // Oœ X

    MoveToEx(hdc, rect.left, rect.top, NULL);
    LineTo(hdc, rect.left, rect.bottom);  // Oœ Y

    // Przesuwanie danych w lewo, aby symulowaæ ruch wykresu
    for (int i = 0; i < width - 1; i++)
    {
        graphData[i] = graphData[i + 1];
    }

    // Generowanie nowej losowej wartoœci na osi Y
    graphData[width - 1] = rand() % (GRAPH_MAX_Y_VALUE + 1);

    // Rysowanie wykresu
    for (int i = 0; i < width - 1; i++)
    {
        int x1 = rect.left + i;
        int y1 = rect.bottom - (graphData[i] * height / GRAPH_MAX_Y_VALUE);

        int x2 = rect.left + i + 1;
        int y2 = rect.bottom - (graphData[i + 1] * height / GRAPH_MAX_Y_VALUE);

        MoveToEx(hdc, x1, y1, NULL);
        LineTo(hdc, x2, y2);
    }

    DeleteObject(hPen);
}

// Procedura obs³ugi okna
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT rect;
        GetClientRect(hwnd, &rect);

        // Pobranie bie¿¹cego czasu w sekundach od pó³nocy
        time_t now = time(0);
        struct tm timeinfo;
        localtime_s(&timeinfo, &now);
        int seconds = timeinfo.tm_hour * 3600 + timeinfo.tm_min * 60 + timeinfo.tm_sec;

        // Rysowanie wykresu
        DrawGraph(hdc, rect, seconds);

        EndPaint(hwnd, &ps);
    }
    break;
    case WM_TIMER:
        InvalidateRect(hwnd, NULL, TRUE);
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
    wc.lpszClassName = (LPCWSTR) g_szClassName;
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
        L"Real-Time Graph (Right to Left)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL)
    {
        MessageBox(NULL, L"Window Creation Failed!", L"Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // Ustawienie timera na odœwie¿anie co sekundê
    SetTimer(hwnd, 1, 1000, NULL);

    // Pêtla wiadomoœci
    while (GetMessage(&Msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    return Msg.wParam;
}
#endif


#if 0
#include <windows.h>
#include <time.h>
#include "framework.h"
#include "WindowsProject2.h"

const char g_szClassName[] = "MyWindowClass";

// Rozmiar okna
#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 400

// Funkcja rysuj¹ca wykres
void DrawGraph(HDC hdc, RECT rect, int second)
{

    OutputDebugStringW(L"test");

    // Wype³nij t³o na bia³o
    FillRect(hdc, &rect, (HBRUSH)(COLOR_WINDOW + 1));

    // Ustawienia pióra (linia wykresu)
    HPEN hPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 255));
    SelectObject(hdc, hPen);

    // Wysokoœæ wykresu
    int height = rect.bottom - rect.top;

    // Rysowanie osi X i Y
    MoveToEx(hdc, rect.left, rect.bottom, NULL);
    LineTo(hdc, rect.right, rect.bottom);  // Oœ X

    MoveToEx(hdc, rect.left, rect.top, NULL);
    LineTo(hdc, rect.left, rect.bottom);  // Oœ Y

    // Rysowanie punktu odpowiadaj¹cego bie¿¹cemu czasowi
    int x = rect.left + (rect.right - rect.left) * second / 86400;
    int y = rect.bottom - height * 0.5;  // Wartoœæ sta³a na 50% wysokoœci

    // Rysowanie punktu
    Ellipse(hdc, x - 3, y - 3, x + 3, y + 3);

    DeleteObject(hPen);
}

// Procedura obs³ugi okna
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT rect;
        GetClientRect(hwnd, &rect);

        // Pobranie bie¿¹cego czasu w sekundach od pó³nocy
        time_t now = time(0);
        struct tm timeinfo;
        localtime_s(&timeinfo, &now);
        int seconds = timeinfo.tm_hour * 3600 + timeinfo.tm_min * 60 + timeinfo.tm_sec;

        // Rysowanie wykresu
        DrawGraph(hdc, rect, seconds);

        EndPaint(hwnd, &ps);
    }
    break;
    case WM_TIMER:
        InvalidateRect(hwnd, NULL, TRUE);
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
        L"Real-Time 24-Hour Graph",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, WINDOW_WIDTH, WINDOW_HEIGHT,
        NULL, NULL, hInstance, NULL);

    if (hwnd == NULL)
    {
        MessageBox(NULL, L"Window Creation Failed!", L"Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // Ustawienie timera na odœwie¿anie co sekundê
    SetTimer(hwnd, 1, 1000, NULL);

    // Pêtla wiadomoœci
    while (GetMessage(&Msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
    }

    return Msg.wParam;
}
#endif


/*
*/
#if 0
// WindowsProject2.cpp : Defines the entry point for the application.
//

#include "framework.h"
#include "WindowsProject2.h"

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
    LoadStringW(hInstance, IDC_WINDOWSPROJECT2, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_WINDOWSPROJECT2));

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
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_WINDOWSPROJECT2));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_WINDOWSPROJECT2);
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