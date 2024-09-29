[BITS 16]               ; We're working in 16-bit real mode
[ORG 0x7C00]           ; BIOS loads bootloader to memory at 0x7C00

start:
    ; Set video mode to 80x25 text mode (mode 3h)
    mov ax, 0x0003     ; 80x25 text mode
    int 0x10           ; Call BIOS video interrupt

    ; Clear the screen
    mov ax, 0xB800     ; Video memory segment for text mode
    mov es, ax         ; ES = video memory segment
    xor di, di         ; DI = 0, start at the top-left of the screen
    mov cx, 2000       ; 80 columns * 25 rows = 2000 characters

clear_screen:
    mov ax, 0x0720     ; Character (space) + attribute (white on black)
    stosw              ; Store word at ES:DI (character + attribute)
    loop clear_screen   ; Repeat until the screen is cleared

    ; Print "Hello, World!" at the top-left corner
    mov si, msg        ; Load address of the message into SI
    mov di, 0          ; Start at the beginning of video memory
    call print_string   ; Call the function to print the string

hang:
    jmp hang           ; Infinite loop to prevent execution from continuing

print_string:
    ; Print a null-terminated string
    mov ah, 0x0E       ; Set teletype output function (int 10h)
print_loop:
    lodsb              ; Load byte from string (SI) into AL
    cmp al, 0          ; Check for null terminator
    je done            ; If AL == 0, we are done
    int 0x10          ; Print character in AL
    jmp print_loop     ; Repeat for the next character
done:
    ret                ; Return from the function

msg db "Hello, World!", 0  ; Null-terminated string

times 510-($-$$) db 0       ; Pad to 510 bytes
dw 0xAA55                   ; Boot signature (0xAA55)
