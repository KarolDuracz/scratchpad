#include <windows.h>
#include <commctrl.h>
#include <string>

#define IDI_ICON 101
#define IDC_PROCESS_LIST 200
#define IDM_FILE_NEW 1001
#define IDM_FILE_EXIT 1000
#define IDM_WINDOW_MANAGE 1002
#define IDM_WINDOW_SETTINGS 1003
#define IDM_WINDOW_CONTROL 1004
#define ID_MDI_CLIENT 1005
#define ID_MDI_FIRSTCHILD 1006
#define ID_PROCESS_CUSTOM_PATH 1007
#define ID_OK_BUTTON 1008
#define ID_CANCEL_BUTTON 1009

const TCHAR ClassName[] = TEXT("MainWindowClass");
const TCHAR ChildClassName[] = TEXT("ChildWindowClass");

HWND hWndClient;       // MDI client window
HWND hProcessDialog;   // Process selection wizard dialog

// Function declarations
void ShowProcessWizard(HWND hwnd);
void EmbedProcess(HWND hwndChild, LPCWSTR appPath);
void EmbedCalculatorInWorkspace(HWND workspace);


// Child window procedure for the MDI child window
LRESULT CALLBACK ChildProc(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam) {
    switch (Msg) {
    case WM_CREATE:
        break;
    case WM_MDIACTIVATE:
        // Placeholder for actions before child activation
        break;
    case WM_SIZE:
        break;
    default:
        return DefMDIChildProc(hWnd, Msg, wParam, lParam);
    }
    return 0;
}

// Dialog procedure for the process selection wizard
LRESULT CALLBACK ProcessDialogProc(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam) {
    switch (Msg) {
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case ID_OK_BUTTON: {
            wchar_t appPath[MAX_PATH];
            HWND hCustomPath = GetDlgItem(hWnd, ID_PROCESS_CUSTOM_PATH);
            GetWindowText(hCustomPath, appPath, MAX_PATH);

            if (wcslen(appPath) > 0) {
                HWND hChild = CreateWindowEx(
                    WS_EX_MDICHILD, ChildClassName, TEXT("Child Window"),
                    WS_CHILD | WS_VISIBLE | WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                    CW_USEDEFAULT, CW_USEDEFAULT, hWndClient, NULL,
                    (HINSTANCE)GetWindowLong(hWnd, GWL_HINSTANCE), NULL);

                if (hChild) {
                    EmbedProcess(hChild, appPath);
                }
            }
            DestroyWindow(hWnd);
            hProcessDialog = NULL;
            break;
        }
        case ID_CANCEL_BUTTON:
            DestroyWindow(hWnd);
            hProcessDialog = NULL;
            break;
        }
        break;

    case WM_CLOSE:
        DestroyWindow(hWnd);
        hProcessDialog = NULL;
        break;

    default:
        return DefWindowProc(hWnd, Msg, wParam, lParam);
    }
    return 0;
}

// Dialog procedure for the process selection wizard
LRESULT CALLBACK ProcessDialogProc2(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam) {
    switch (Msg) {

    case WM_MOVE:
        OutputDebugStringA("xx");
        break;

    case WM_COMMAND:   
        switch (LOWORD(wParam)) {
            case ID_OK_BUTTON: {
                OutputDebugStringA("11");
                EmbedCalculatorInWorkspace(hWnd);
            }
        }
        break;

    case WM_CLOSE:
        DestroyWindow(hWnd);
        hProcessDialog = NULL;
        break;

    default:
        return DefWindowProc(hWnd, Msg, wParam, lParam);
    }
    return 0;
}


// Function to show the process selection wizard dialog
void ShowProcessWizard(HWND hwnd) {
    //if (hProcessDialog) return;  // Ensure only one instance

    OutputDebugStringA("test");

    HWND hChild;
    CREATESTRUCT cs;
    ZeroMemory(&cs, sizeof(CREATESTRUCT));
    hChild = CreateWindowEx(WS_EX_MDICHILD, ChildClassName, TEXT("Child Window"), WS_CHILD | WS_VISIBLE | WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, hWndClient, NULL, (HINSTANCE)GetWindowLong(hwnd, GWL_HINSTANCE), &cs);
    

    SetWindowLongPtr(hChild, GWLP_WNDPROC, (LONG_PTR)ProcessDialogProc2);


    // OK and Cancel buttons
    CREATESTRUCT btn_cs;
    ZeroMemory(&btn_cs, sizeof(CREATESTRUCT));
    CreateWindow(TEXT("BUTTON"), TEXT("OK"), WS_CHILD | WS_VISIBLE,
        50, 120, 80, 30, hChild, (HMENU)ID_OK_BUTTON, NULL, &btn_cs);

    /*
    CreateWindow(
        TEXT("EDIT"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
        20, 40, 240, 20, hChild, (HMENU)ID_PROCESS_CUSTOM_PATH, NULL, NULL);

    // OK and Cancel buttons
    CreateWindow(TEXT("BUTTON"), TEXT("OK"), WS_CHILD | WS_VISIBLE,
        50, 120, 80, 30, hChild, (HMENU)ID_OK_BUTTON, NULL, NULL);
    CreateWindow(TEXT("BUTTON"), TEXT("Cancel"), WS_CHILD | WS_VISIBLE,
        150, 120, 80, 30, hChild, (HMENU)ID_CANCEL_BUTTON, NULL, NULL);
    */

    if (!hChild)
        MessageBox(hwnd, TEXT("Failed To Create The Child Window"), TEXT("Error"), MB_OK);
   
}

void EmbedCalculatorInWorkspace(HWND workspace) {
    PROCESS_INFORMATION pi;
    STARTUPINFO si;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // Start the Calculator process
    if (CreateProcess(
        //L"C:\\Windows\\System32\\calc.exe", // Path to calc.exe
        //L"C:\\Program Files (x86)\\Windows Media Player\\wmplayer.exe",
       // L"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
       // L"C:\\Windows\\System32\\notepad.exe",
       // L"C:\\Windows\\System32\\cmd.exe",
        L"C:\\Windows\\System32\\mspaint.exe",
        NULL,   // Command line arguments
        NULL,   // Process handle not inheritable
        NULL,   // Thread handle not inheritable
        FALSE,  // Set handle inheritance to FALSE
        0,      // No creation flags
        NULL,   // Use parent's environment block
        NULL,   // Use parent's starting directory 
        &si,    // Pointer to STARTUPINFO structure
        &pi)    // Pointer to PROCESS_INFORMATION structure
        ) {
        // Wait until the Calculator window is created
        Sleep(1000); // You might want to use a better way to ensure the window is ready

        // Get the handle of the Calculator's main window
       // HWND hCalcWnd = FindWindow(NULL, L"Windows PowerShell");
      //  HWND hCalcWnd = FindWindow(NULL, L"Untitled - Notepad");
       // HWND hCalcWnd = FindWindow(NULL, L"C:\\Windows\\System32\\cmd.exe");
        HWND hCalcWnd = FindWindow(NULL, L"Untitled - Paint");
        if (hCalcWnd) {
            SetParent(hCalcWnd, workspace); // Set the parent to the workspace
            // Resize and reposition the calculator window to fit inside the workspace
            RECT rc;
            GetClientRect(workspace, &rc);
            SetWindowPos(hCalcWnd, NULL, 0, 0, rc.right, rc.bottom, SWP_NOZORDER);
            OutputDebugStringA("hwnd is connected");
        }

        // get window name
        //GetWindowTextA((HWND)pi.hProcess, &out, sizeof()
        int ret = GetWindowTextLengthA((HWND)pi.hProcess);
        OutputDebugStringA(std::to_string(ret).c_str());
        OutputDebugStringA("testr \n\n");
        //OutputDebugStringA((LPCSTR)si.lpTitle);

        // Close handles to the process
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }

}

// Function to embed a specified process within an MDI child window
void EmbedProcess(HWND hwndChild, LPCWSTR appPath) {
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_SHOWNORMAL;

    if (!CreateProcess(appPath, NULL, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi)) {
        MessageBox(NULL, L"Failed to start process.", L"Error", MB_OK | MB_ICONERROR);
    }
}

// Main window procedure
LRESULT CALLBACK WndProc(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam) {
    switch (Msg) {
    case WM_CREATE: {
        HMENU hMenubar = CreateMenu();
        HMENU hMenuFile = CreateMenu();
        HMENU hMenuWindow = CreateMenu();

        AppendMenu(hMenuFile, MF_STRING, IDM_FILE_NEW, TEXT("New"));
        AppendMenu(hMenuFile, MF_STRING, IDM_FILE_EXIT, TEXT("Quit"));
        AppendMenu(hMenubar, MF_POPUP, (UINT_PTR)hMenuFile, TEXT("&File"));

        AppendMenu(hMenuWindow, MF_STRING, IDM_WINDOW_MANAGE, TEXT("Manage"));
        AppendMenu(hMenuWindow, MF_STRING, IDM_WINDOW_SETTINGS, TEXT("Settings"));
        AppendMenu(hMenuWindow, MF_STRING, IDM_WINDOW_CONTROL, TEXT("Control"));
        AppendMenu(hMenubar, MF_POPUP, (UINT_PTR)hMenuWindow, TEXT("&Window"));

        SetMenu(hWnd, hMenubar);

        CLIENTCREATESTRUCT ccs;
        ccs.hWindowMenu = GetSubMenu(GetMenu(hWnd), 0);
        ccs.idFirstChild = ID_MDI_FIRSTCHILD;

        hWndClient = CreateWindowEx(WS_EX_CLIENTEDGE, TEXT("MDICLIENT"), NULL,
            WS_CHILD | WS_CLIPCHILDREN | WS_VSCROLL | WS_HSCROLL,
            CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
            hWnd, (HMENU)ID_MDI_CLIENT, (HINSTANCE)GetWindowLong(hWnd, GWL_HINSTANCE), &ccs);

        if (!hWndClient) {
            MessageBox(hWnd, TEXT("Failed To Create The Client Window"), TEXT("Error"), MB_OK);
        }
        ShowWindow(hWndClient, SW_SHOW);
        break;
    }

    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDM_FILE_NEW:
            ShowProcessWizard(hWnd);
            break;

        case IDM_FILE_EXIT:
            PostMessage(hWnd, WM_CLOSE, 0, 0);
            break;

        case IDM_WINDOW_MANAGE:
            MessageBox(hWnd, TEXT("Window Management Placeholder"), TEXT("Manage"), MB_OK);
            break;

        case IDM_WINDOW_SETTINGS:
            MessageBox(hWnd, TEXT("Settings Placeholder"), TEXT("Settings"), MB_OK);
            break;

        case IDM_WINDOW_CONTROL:
            MessageBox(hWnd, TEXT("Control Placeholder"), TEXT("Control"), MB_OK);
            break;

        default:
            if (LOWORD(wParam) >= ID_MDI_FIRSTCHILD) {
                DefFrameProc(hWnd, hWndClient, Msg, wParam, lParam);
            }
            else {
                HWND hChild = (HWND)SendMessage(hWndClient, WM_MDIGETACTIVE, 0, 0);
                if (hChild) SendMessage(hChild, WM_COMMAND, wParam, lParam);
            }
        }
        return 0;

    case WM_CLOSE:
        DestroyWindow(hWnd);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefFrameProc(hWnd, hWndClient, Msg, wParam, lParam);
    }
    return 0;
}

// WinMain entry point
INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, INT nCmdShow) {
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_HREDRAW | CS_VREDRAW, WndProc, 0, 0, hInstance,
                      LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON)), LoadCursor(NULL, IDC_ARROW),
                      (HBRUSH)(COLOR_WINDOW + 1), NULL, ClassName, LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON)) };

    if (!RegisterClassEx(&wc)) {
        MessageBox(NULL, TEXT("Failed To Register The Window Class."), TEXT("Error"), MB_OK | MB_ICONERROR);
        return 0;
    }

    wc.lpfnWndProc = ChildProc;
    wc.lpszClassName = ChildClassName;

    // Register child window class
    if (!RegisterClassEx(&wc)) {
        MessageBox(NULL, TEXT("Failed To Register The Child Window Class"), TEXT("Error"), MB_OK | MB_ICONERROR);
        return 0;
    }

    // Create the main window
    HWND hWnd = CreateWindowEx(WS_EX_CLIENTEDGE, ClassName, TEXT("MDI Application"),
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 600, 400,
        NULL, NULL, hInstance, NULL);

    if (!hWnd) {
        MessageBox(NULL, TEXT("Window Creation Failed."), TEXT("Error"), MB_OK | MB_ICONERROR);
        return 0;
    }

    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    // Message loop
    MSG Msg;
    while (GetMessage(&Msg, NULL, 0, 0)) {
        if (!TranslateMDISysAccel(hWndClient, &Msg)) {
            TranslateMessage(&Msg);
            DispatchMessage(&Msg);
        }
    }

    return Msg.wParam;
}
