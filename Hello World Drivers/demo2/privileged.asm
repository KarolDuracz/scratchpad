.code

; Exported function to execute CLI instruction
PUBLIC DoCli
DoCli PROC
    cli         ; Clear interrupt flag
    ret
DoCli ENDP

END
