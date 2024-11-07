@echo off
setlocal

rem Set the paths to the WDK tools
set WDK_INC=C:\Program Files (x86)\Windows Kits\8.0\Include\km
set WDK_INC2=C:\Program Files (x86)\Windows Kits\8.0\Include\km\crt
set WDK_LIB=C:\Program Files (x86)\Windows Kits\8.0\Lib\win8\km\x64

echo WDK_INC is %WDK_INC%

echo WDK_INC2 is %WDK_INC2%


rem Compile the driver
CL.exe /c /W4 /D "_AMD64_" /D "WINVER=0x0603" /D "NTDDI_VERSION=NTDDI_WINBLUE" /I "%WDK_INC%" /I "%WDK_INC2%" helloworld_kernel.c /FeHelloWorld.obj

rem Link the driver
link.exe /driver helloworld_kernel.obj /OUT:HelloWorld.sys /LIBPATH:"%WDK_LIB%" ntoskrnl.lib /ENTRY:DriverEntry /SUBSYSTEM:NATIVE
endlocal
