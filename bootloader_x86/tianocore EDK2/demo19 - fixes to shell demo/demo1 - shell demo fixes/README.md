
<h3>How to use (example) - capmem command</h3>

```capmem start 5000 512``` — allocates 5,000 slots × 512 chars/line (capped by the safety limits).

```capmem status``` — see allocation & activity.

```capmem stop``` — stop writing new lines (you can still capmem start again to resume, or keep the buffer).

Run loadimg or other program — all OutputString/Print text will be copied into the memory slots as it arrives.

```capmem save``` — writes the captured lines to \EFI\Boot\myLogs\log-...txt.

```capmem free``` — free the capture buffer.

Added log collection mechanism from 12.4.3. EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL.OutputString() https://uefi.org/specs/UEFI/2.10/12_Protocols_Console_Support.html#efi-simple-text-output-protocol-outputstring - In short, it collects logs sent to OutputString from functions like Print and others that display text on the console using this protocol. This allows to launch applications with the "loadimg" command, and all logs generated from those applications will be in this buffer.
<br /><br />
Simply start with the command "capmem start 5000 512" . It can run in the background, and you can continue using this shell as I showed in demo #17. And these logs will be collected for this temporary buffer. We can then check the status with the command "capmem status". Sometimes you need to turn it off, e.g. when you want to use the editor to check the logs directly from this shell. Because it collects logs all the time and then it takes the whole buffer and suddenly it becomes 3000, 5000 lines and more when you scroll up and down the pages in the editor. That's why there is a stop order ```capmem stop``` . Then you can resume it again, e.g. by setting the same buffer parameters again: capmem start 5000 512.
<br /><br />
