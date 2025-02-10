.DATA
    PUBLIC _vartest         ; Make _vartest a public symbol
    _vartest QWORD 1234h    ; Define _vartest as a 64-bit variable initialized to 0x1234
	
	PUBLIC _testVMX
	_testVMX QWORD 0h

.DATA?                    ; Uninitialized (BSS) Section
    PUBLIC vmxon_region
    PUBLIC vmcs_region
    vmxon_region QWORD 1024 DUP (?)  ; 4KB aligned VMXON region
    vmcs_region QWORD 1024 DUP (?)   ; 4KB aligned VMCS region

.CODE

    PUBLIC DoCli            ; Export the DoCli function
DoCli PROC
    cli                     ; Clear the interrupt flag
    PUSHFQ                  ; Push the flags register onto the stack
    POP RAX                 ; Pop the flags into RAX
    MOV [_vartest], RAX       ; Store the value of RAX in _vartest
    RET
DoCli ENDP

	PUBLIC StartVMX
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
	
StartVMX ENDP

	PUBLIC checkVMX
checkVMX PROC
; -------------------------------
; StartVMX - Test and Enable VMX
; -------------------------------
	cli
    MOV RCX, 3ah
	RDMSR
	MOV [_testVMX], RAX
	sti
    ret
	
checkVMX ENDP

END
