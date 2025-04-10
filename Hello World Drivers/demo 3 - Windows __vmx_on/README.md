<h2>Quick intro to VMX feature and run VM on ring 1</h2>
But this is only simple test, to enable VMX feature from Windows Driver Kernel perspective, and check if it is possible to manipulate from this level. This is not complex implementation to demonstrate VMX in core. This is only simple test and introduce to check if __vmx_on function run properly. <br />
This is continuation for previous demo 1 and 2. <br />
NOTE. This allocates memory in PID 4 (system process) And turns on VMX on VCR4. 

Part for enable VMX feature on CR4. Even though Hyper-VT is enabled in Windows, it allows you to execute the driver code as shown in the image.
```
StartVMX PROC
; -------------------------------
; StartVMX - Test and Enable VMX
; -------------------------------
	cli
    MOV RCX, 3ah
	RDMSR
	MOV [_testVMX], RAX
	; enable VMX on CR4
	mov     rax, cr4
	bts     rax, 13          ; Set CR4.VMXE (bit 13)
	mov     cr4, rax
	sti
  ret
```

__vmx_on is on line 62 https://github.com/KarolDuracz/scratchpad/blob/main/Hello%20World%20Drivers/demo%203%20-%20Windows%20__vmx_on/PrivilegedInstructionsDriver.c#L62

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/Hello%20World%20Drivers/demo%203%20-%20Windows%20__vmx_on/22%20-%2002-02-2025%20-%20chyba%20dziala.png)

This requires deeper analysis and explanation, but that's all for now.
<br /><br />
One important thing I have to write is that the value 0x5 from the listing "test VMX value at:" shows that VMX is not available, and you need to enable bit 13 on CR4. Otherwise you will get a blue screen. Crash system error. (...) Although I don't know if it actually matters, because after enabling bit 13 I still have 0x5 in this MSR register. But definitely bit 13 on CR4 is needed to avoid blue screen. At least for me it started working after setting this...

<br /><br />
References:<br />
https://learn.microsoft.com/en-us/cpp/intrinsics/vmx-on?view=msvc-170 <br />
https://rayanfam.com/topics/hypervisor-from-scratch-part-3/ <br />
https://wiki.osdev.org/VMX <br />
https://alice.climent-pommeret.red/posts/a-syscall-journey-in-the-windows-kernel/ <-- this is next step for this issue
