// compile with /link user32.lib
// cl simple_demo.c /link User32.lib

#include <Windows.h>
#include <stdio.h>

int main()
{
	MessageBoxA(NULL, L"hello", L"test", MB_OK);
	while(1);
}