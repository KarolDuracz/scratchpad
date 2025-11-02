I will skip the long description this time. I'll just show how it works on images, what I changed in these demos.
<br /><br />
<b>1. Minimalist Shell FIXED AND IMPROVED.</b> - I don't need to take photos anymore. Currently, the repo is over ~600MB - mostly raw photos. This gives the ability to collect logs and save them to a USB flash drive.
<br /><br />
https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo1%20-%20shell%20demo%20fixes
<br /><br />
<b>2. Simple text editor for logs ( read only )</b> - In order to not exit from shell for reading logs, but read them in this mode, in shell mode.
<br /><br />
https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo2%20-%20log%20reader
<br /><br />
<b>3. GOP fun - playing with custom fonts and image loading, simple pointer protocol, etc.</b> - This is not integrated with this demo shell. It runs as "loadimg helloworld.efi" from this shell only. This is just to show some of the possibilities offered by GOP and the default protocol. Even at 1024x768 resolution, you can see that these images look quite good. Simply to play with it more in free time. But GUI isn't what I want to focus on here.
<br /><br />
https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo19%20-%20fixes%20to%20shell%20demo/demo3%20-%20gop%20fun
<br /><br /><br />

<hr>

> [!NOTE]
> How to build demos from source code - I build it on EmulatorPkg ( helloworld ) and copy helloworld.efi

Every demo from 1-19 (up to now) is built on top of HelloWorld.c https://github.com/tianocore/edk2/tree/master/MdeModulePkg/Application/HelloWorld . So I build it with the commands. But you could also build it with EmulatorPkg and copy helloworld.efi ( in demo 18 there is a .bat script that does this ) 

```
// main edk2 folder, go here first
cd edk2

// setup environment - read guide for edk2 installation and configuration which I wrote here
edksetup.bat

// compile via .inf just like that
build -p MdeModulePkg/MdeModulePkg.dsc -m MdeModulePkg/Application/HelloWorld/HelloWorld.inf
```

<h2>Some information for summary of demos 0 - 19 after 1 year of learning it</h2>

When you go to main folder here -> https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2 and you will start doing the same thing as I started a year ago. The first thing you need to pay attention to ( notice ), I'm still use ```git checkout tags/edk2-stable202411``` version. On Win 8.1. 

```
git clone https://github.com/tianocore/edk2.git
cd edk2
git submodule update --init
git tag
git checkout tags/edk2-stable202411
```

Now, current stable version is different - current state for 02-11-2025.

```
https://github.com/tianocore/edk2/releases/tag/edk2-stable202508
```

Here's a /Conf/target.txt file which sets the project for compilation - Currently I just compile the whole EmulatorPkg and copy 1 file ```helloworld.efi``` from it. Until to this point I haven't used anything else that these 2 packs in ACTIVE_PLATFORM ( EmulatorPkg / OvmfPkgX64 ). You also see that it's X64 all the time. DEBUG.
<br /><br />
So I just do this, setup edksetup and ```build``` the whole EmulatorPkg project

```
edksetup.bat
build // just build whole project EmulatorPkg and copy helloworld.efi from after compilation
```

/Conf/target.txt

```
#
#  Copyright (c) 2006 - 2019, Intel Corporation. All rights reserved.<BR>
#
#  SPDX-License-Identifier: BSD-2-Clause-Patent
#
#
#  ALL Paths are Relative to WORKSPACE

#  Separate multiple LIST entries with a SINGLE SPACE character, do not use comma characters.
#  Un-set an option by either commenting out the line, or not setting a value.

#
#  PROPERTY              Type       Use         Description
#  ----------------      --------   --------    -----------------------------------------------------------
#  ACTIVE_PLATFORM       Filename   Recommended Specify the WORKSPACE relative Path and Filename
#                                               of the platform description file that will be used for the
#                                               build. This line is required if and only if the current
#                                               working directory does not contain one or more description
#                                               files.
ACTIVE_PLATFORM       = EmulatorPkg/EmulatorPkg.dsc
#ACTIVE_PLATFORM 	   = OvmfPkg/OvmfPkgX64.dsc

#  TARGET                List       Optional    Zero or more of the following: DEBUG, RELEASE, NOOPT
#                                               UserDefined; separated by a space character.
#                                               If the line is missing or no value is specified, all
#                                               valid targets specified in the platform description file
#                                               will attempt to be built. The following line will build
#                                               DEBUG platform target.
TARGET                = DEBUG

#  TARGET_ARCH           List       Optional    What kind of architecture is the binary being target for.
#                                               One, or more, of the following, IA32, IPF, X64, EBC, ARM
#                                               or AArch64.
#                                               Multiple values can be specified on a single line, using
#                                               space characters to separate the values.  These are used
#                                               during the parsing of a platform description file,
#                                               restricting the build output target(s.)
#                                               The Build Target ARCH is determined by (precedence high to low):
#                                                 Command-line: -a ARCH option
#                                                 target.txt: TARGET_ARCH values
#                                                 DSC file: [Defines] SUPPORTED_ARCHITECTURES tag
#                                               If not specified, then all valid architectures specified
#                                               in the platform file, for which tools are available, will be
#                                               built.
TARGET_ARCH           = X64

#  TOOL_DEFINITION_FILE  Filename  Optional   Specify the name of the filename to use for specifying
#                                             the tools to use for the build.  If not specified,
#                                             WORKSPACE/Conf/tools_def.txt will be used for the build.
TOOL_CHAIN_CONF       = Conf/tools_def.txt

#  TAGNAME               List      Optional   Specify the name(s) of the tools_def.txt TagName to use.
#                                             If not specified, all applicable TagName tools will be
#                                             used for the build.  The list uses space character separation.
TOOL_CHAIN_TAG        = VS2019

# MAX_CONCURRENT_THREAD_NUMBER  NUMBER  Optional  The number of concurrent threads. If not specified or set
#                                                 to zero, tool automatically detect number of processor
#                                                 threads. Recommend to set this value to one less than the
#                                                 number of your computer cores or CPUs. When value set to 1,
#                                                 means disable multi-thread build, value set to more than 1,
#                                                 means user specify the thread number to build. Not specify
#                                                 the default value in this file.
# MAX_CONCURRENT_THREAD_NUMBER = 1


# BUILD_RULE_CONF  Filename Optional  Specify the file name to use for the build rules that are followed
#                                     when generating Makefiles. If not specified, the file:
#                                     WORKSPACE/Conf/build_rule.txt will be used
BUILD_RULE_CONF = Conf/build_rule.txt
```

