// Filename: HookDLL.c
#include <windows.h>

HHOOK hMouseHook;
HHOOK hKeyboardHook;
HINSTANCE hInstance;
int ctrlPressed = 0;

// Clipboard function to set data
void SetClipboardText(const char *text) {
    if (OpenClipboard(NULL)) {
        EmptyClipboard();
        HGLOBAL hGlobal = GlobalAlloc(GMEM_MOVEABLE, strlen(text) + 1);
        if (hGlobal) {
            memcpy(GlobalLock(hGlobal), text, strlen(text) + 1);
            GlobalUnlock(hGlobal);
            SetClipboardData(CF_TEXT, hGlobal);
        }
        CloseClipboard();
    }
}

// Mouse hook procedure
LRESULT CALLBACK MouseProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0 && wParam == WM_LBUTTONDOWN && ctrlPressed) {
        // CTRL + Left mouse click detected
        SetClipboardText("Data copied from service!"); // Example text
        MessageBox(NULL, "Data copied to clipboard", "Clipboard Service", MB_OK);
    }
    return CallNextHookEx(hMouseHook, nCode, wParam, lParam);
}

// Keyboard hook procedure
LRESULT CALLBACK KeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        if (wParam == VK_CONTROL) {
            KBDLLHOOKSTRUCT *p = (KBDLLHOOKSTRUCT *)lParam;
            ctrlPressed = (p->flags & LLKHF_UP) ? 0 : 1;
        }
    }
    return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
}

// DLL entry point
BOOL APIENTRY DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    if (fdwReason == DLL_PROCESS_ATTACH) {
        hInstance = hinstDLL;
    }
    return TRUE;
}

// Exported function to set hooks
__declspec(dllexport) void SetHooks() {
    hMouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseProc, hInstance, 0);
    hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardProc, hInstance, 0);
}

// Exported function to remove hooks
__declspec(dllexport) void RemoveHooks() {
    UnhookWindowsHookEx(hMouseHook);
    UnhookWindowsHookEx(hKeyboardHook);
}
