<h2>EDK2 Hello World extended demo for EmulatorPkg to analyze internal structures and functions</h2>

I found an interesting repo that has a list of many lessons https://github.com/Kostr/UEFI-Lessons/tree/master

Applies to these files (in demo 1 and 2 I also changed HelloWord instead of creating a new module)
https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/HelloWorld

Source
HelloWorld.c
HelloWorld.inf

Build
build -p MdeModulePkg/MdeModulePkg.dsc -m MdeModulePkg/Application/HelloWorld/HelloWorld.inf
