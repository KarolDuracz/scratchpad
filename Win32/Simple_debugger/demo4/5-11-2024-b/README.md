NOT FIXED <br />
you must change<br />
In debugger.cpp in line 97 : char ret[64]; to >>>> unsigned char ret[64]; <br />
This means, form type ```char``` to ```unsigned char``` !
<br /><br />
fixed
<br /><br />
1. Add PID as window title in SimpleGUI.cpp ```SetWindowTextA(hwnd, std::to_string(GetProcessId(GetCurrentProcess())).c_str());``` <br >
2. Add some stuff to debugger.cpp but still not working well.
