; bootloader.asm - A simple bootloader that prints "Hello World!" on the screen
; and hangs. It should be compiled with NASM and can be loaded using an emulator
; or burned to a bootable medium (like USB).

BITS 16                ; We are working in 16-bit real mode

start:
    ; Clear the screen by setting the video memory to all spaces (0x20)
    mov ax, 0xB800      ; Video memory segment (text mode)
    mov es, ax          ; ES = Video memory segment
    xor di, di          ; DI = 0, so we write to the start of video memory
    mov cx, 2000        ; Clear the first 2000 bytes (80x25 screen)

clear_loop:
    mov al, 0x20        ; ' ' (space character)
    mov ah, 0x07        ; White text on black background
    stosw               ; Write AL and AH (character and attribute) to memory
    loop clear_loop     ; Repeat until screen is clear

    ; Print "Hello World!" on the screen at the top left (position 0)
    mov si, msg         ; SI = address of the message
    mov di, 0           ; DI = 0 (start of video memory)
    
    call print_string   ; Print the string

hang:
    jmp hang            ; Infinite loop to prevent bootloader from continuing

print_string:
    ; Print a null-terminated string
    mov ah, 0x0E        ; BIOS teletype function for text output (int 10h)
print_loop:
    lodsb               ; Load byte from string (SI) to AL
    cmp al, 0           ; Is it the null terminator?
    je done             ; If yes, we're done printing
    int 0x10            ; Otherwise, print character in AL
    jmp print_loop      ; Repeat for next character
done:
    ret                 ; Return to caller

msg db "Hello World!", 0  ; Null-terminated string

times 510-($-$$) db 0     ; Pad with zeros until byte 510
dw 0xAA55                 ; Boot signature
