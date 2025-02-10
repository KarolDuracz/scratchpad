@echo off
setlocal

rem Set the paths to the WDK tools
set WDK_INC=C:\Program Files (x86)\Windows Kits\8.0\Include\km
set WDK_INC_CRT=C:\Program Files (x86)\Windows Kits\8.0\Include\km\crt
set WDK_LIB=C:\Program Files (x86)\Windows Kits\8.0\Lib\win8\km\x64
set WDK_LIB_CRT=C:\Program Files (x86)\Windows Kits\8.0\Lib\win8\um\x64
set WDK_BIN=C:\Program Files (x86)\Windows Kits\8.0\bin\x64
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\lib\x64

echo WDK_INC is %WDK_INC%
echo WDK_LIB is %WDK_LIB%
echo WDK_LIB_CRT is %WDK_LIB_CRT%
echo MSVC is %MSVC%

rem Compile the driver
CL.exe /c /W4 /D "_AMD64_" /D "WINVER=0x0603" /D "NTDDI_VERSION=NTDDI_WINBLUE" /D "NT_KERNEL_MODE" /D "_KERNEL_MODE" /GS- /I "%WDK_INC%" /I "%WDK_INC_CRT%" PrivilegedInstructionsDriver.c /FePrivilegedInstructionsDriver.obj

rem Assemble the privileged instructions file
ml64.exe /c privileged.asm /Fo:privileged.obj

rem Link the driver
link.exe /driver PrivilegedInstructionsDriver.obj privileged.obj /OUT:PrivilegedInstructionsDriver15.sys /LIBPATH:"%WDK_LIB%" /LIBPATH:"%WDK_LIB_CRT%" /LIBPATH:"%MSVC%" ntoskrnl.lib /ENTRY:DriverEntry /SUBSYSTEM:NATIVE /NODEFAULTLIB:LIBCMT.lib

endlocal
