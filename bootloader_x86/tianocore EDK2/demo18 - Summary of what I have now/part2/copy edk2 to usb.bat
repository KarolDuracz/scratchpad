@echo off
color a
rem ----- paths -----
set "SRC=C:\Users\kdhome\Documents\progs\edk2_win81\edk2\Build\EmulatorX64\DEBUG_VS2019\X64"
set "DST=F:\EFI\Boot\myApps"

echo Source: "%SRC%"
echo Destination: "%DST%"
echo.

rem create destination if missing
if not exist "%DST%" (
    echo Destination not found — creating "%DST%"...
    mkdir "%DST%" 2>nul
    if errorlevel 1 (
        echo Failed to create destination "%DST%".
        pause
        exit /b 1
    )
)

rem change to source directory
pushd "%SRC%" 2>nul || (
    echo Failed to change directory to "%SRC%".
    pause
    exit /b 1
)

rem ensure source file exists
if not exist "helloworld.efi" (
    echo Source file "helloworld.efi" not found in "%SRC%".
    popd
    pause
    exit /b 1
)

rem copy file (overwrite if exists)
echo Copying helloworld.efi -> "%DST%"\helloworld.efi
copy /Y "helloworld.efi" "%DST%\helloworld.efi" >nul
if errorlevel 1 (
    echo Copy failed.
) else (
    echo Copy succeeded.
)

rem return to previous dir and keep window open
popd
cmd /k
