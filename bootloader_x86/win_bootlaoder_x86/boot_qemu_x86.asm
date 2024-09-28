[BITS 16]
[ORG 0x7C00]

start:
    mov si, msg              ; Point to the message
    call print_string         ; Print the message

hang:
    cli                      ; Disable interrupts
    hlt                      ; Halt the CPU (endless loop)
    
print_string:
    mov ah, 0x0E              ; Teletype output function (BIOS service)
.repeat:
    lodsb                    ; Load next byte of the string
    test al, al               ; Check if the byte is zero (end of string)
    jz .done                 ; If zero, we're done
    int 0x10                 ; Print character
    jmp .repeat              ; Repeat for next character
.done:
    ret                      ; Return to caller

msg db "Hello, World!", 0

times 510-($-$$) db 0         ; Pad with zeros to make 512 bytes (boot sector size)
dw 0xAA55                    ; Boot sector signature
