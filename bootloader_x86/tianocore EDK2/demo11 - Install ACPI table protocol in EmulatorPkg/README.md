<h2>Demo 11 - Install ACPI table protocol in EmulatorPkg</h2>

In the previous demo (demo 10 https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2/demo10%20-%20hello%20world%20extended), I linked to an 
interesting repo with many lessons (https://github.com/Kostr/UEFI-Lessons/tree/master). Someone did a good job. Even in the days of GPT Chat, you need documentation and relevant data to ask the right questions that 
will give you a solution or idea.
<br /><br />
I've looked at a few that could easily be implemented on "HELLO WORLD EXTENDED" from demo10 in EmulatorPkg, but one is interesting. I've been wanting to figure this out 
for a few months now. It's about "Lesson 37: Investigate ways how to add acpiview command functionality to your shell" https://github.com/Kostr/UEFI-Lessons/tree/master/Lessons/Lesson_37
<br /><br />
By default, ACPI VIEW APP is in the EDK2 folder in the shell module but by default it is not included in the EmulatorPkg project and compiled https://github.com/tianocore/edk2/tree/master/ShellPkg/Application/AcpiViewApp

<h3>Demo</h3>

