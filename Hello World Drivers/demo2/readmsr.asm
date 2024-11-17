.code

; Exported function to execute CLI instruction
PUBLIC ReadMSR
ReadMSR PROC
    mov ecx, 0xC0000080
	rdmsr
	mov     [msrValue], rdx
	mov     [msrValue+4], rax
	ret
ReadMSR ENDP

END
