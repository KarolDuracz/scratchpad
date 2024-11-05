// SimpleGUI.cpp
#include <windows.h>
#include <commctrl.h> // For Tab Control
#include <windowsx.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Comctl32.lib") // Link against Comctl32.lib for tab control support

// Constants for menu items
#define ID_MENU_ITEM1 1
#define ID_MENU_ITEM2 2
#define ID_MENU_ITEM3 3

// Global variable for the tab control handle
HWND hTabControl;

// Window Procedure function
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        {
            // Initialize common controls for the tab control
            INITCOMMONCONTROLSEX icex;
            icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
            icex.dwICC = ICC_TAB_CLASSES;
            InitCommonControlsEx(&icex);

            // Create the tab control
            hTabControl = CreateWindowEx(0, WC_TABCONTROL, L"",
                WS_CHILD | WS_VISIBLE | TCS_FOCUSONBUTTONDOWN,
                10, 10, 200, 30, hwnd, NULL, (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE), NULL);

            // Add initial tabs
            TCITEM tie;
            tie.mask = TCIF_TEXT;

            tie.pszText = const_cast<LPWSTR>(L"Tab 1");
            TabCtrl_InsertItem(hTabControl, 0, &tie);

            tie.pszText = const_cast<LPWSTR>(L"Tab 2");
            TabCtrl_InsertItem(hTabControl, 1, &tie);
        }
        break;

    case WM_CONTEXTMENU:
        {
            // Create and display a context menu on right-click
            HMENU hPopupMenu = CreatePopupMenu();
            AppendMenu(hPopupMenu, MF_STRING, ID_MENU_ITEM1, L"Menu Item 1");
            AppendMenu(hPopupMenu, MF_STRING, ID_MENU_ITEM2, L"Menu Item 2");
            AppendMenu(hPopupMenu, MF_STRING, ID_MENU_ITEM3, L"Menu Item 3");

            // Display the popup menu
            TrackPopupMenu(hPopupMenu, TPM_RIGHTBUTTON, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), 0, hwnd, NULL);
            DestroyMenu(hPopupMenu);
        }
        break;

    case WM_COMMAND:
        // Handle menu item selection
        switch (LOWORD(wParam)) {
        case ID_MENU_ITEM1:
            MessageBox(hwnd, L"You selected Menu Item 1", L"Menu", MB_OK);
            break;
        case ID_MENU_ITEM2:
            MessageBox(hwnd, L"You selected Menu Item 2", L"Menu", MB_OK);
            break;
        case ID_MENU_ITEM3:
            MessageBox(hwnd, L"You selected Menu Item 3", L"Menu", MB_OK);
            break;
        }
        break;

    case WM_LBUTTONDOWN:
        {
            // Move tab control horizontally when left-clicked
            static int tabPosX = 10;
            tabPosX += 10; // Move by 10 pixels each click

            // Limit movement within window bounds
            RECT rect;
            GetClientRect(hwnd, &rect);
            if (tabPosX + 200 > rect.right) {
                tabPosX = 10; // Reset position if it goes out of bounds
            }
            SetWindowPos(hTabControl, NULL, tabPosX, 10, 200, 30, SWP_NOZORDER | SWP_NOSIZE);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR /*pCmdLine*/, int nCmdShow) {
    // Register the window class
    const wchar_t CLASS_NAME[] = L"Sample Window Class";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(
        0,                              // Optional window styles
        CLASS_NAME,                     // Window class
        L"Simple GUI Application with Context Menu and Tab Control", // Window title
        WS_OVERLAPPEDWINDOW,            // Window style
        CW_USEDEFAULT, CW_USEDEFAULT, 500, 400, // Window size
        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    // Run the message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
