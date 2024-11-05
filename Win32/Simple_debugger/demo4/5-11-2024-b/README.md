fixed
<br /><br />
1. Add PID as window title in SimpleGUI.cpp ```SetWindowTextA(hwnd, std::to_string(GetProcessId(GetCurrentProcess())).c_str());``` <br >
2. Add some stuff to debugger.cpp but still not working well.
