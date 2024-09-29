BITS 16                   ; We are in 16-bit real mode
ORG 0x7C00                ; Boot sector loads at 0x7C00 by BIOS

start:
    ; Clear registers
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00         ; Set stack pointer

    ; Print message to screen (for debugging)
    mov si, boot_msg
    call print_string

    ; Select the drive (0x01 for `ide1-cd0`, as per QEMU config)
    mov dl, 0x01           ; DL = 0x01 for secondary IDE (CD-ROM)

    ; Load kernel from CD-ROM (assume it's at LBA 0x10 on CD)
    mov bx, 0x9000         ; BX = Load the kernel at 0x9000 (arbitrary memory location)
    mov si, 0x10           ; SI = LBA = 0x10 (sector to load from)
    call read_sector       ; Call read sector function

    ; Jump to loaded kernel
    jmp 0x9000             ; Jump to where we loaded the kernel

hang:
    jmp hang               ; If something goes wrong, loop forever

; Function: read_sector
; Reads one sector (512 bytes) from the CD-ROM (using LBA addressing)
read_sector:
    push ax
    push bx
    push cx
    push dx

    ; Convert LBA (in SI) to CHS for INT 13h
    mov ax, si              ; AX = LBA
    xor dx, dx              ; DX = 0 (clear upper bits)
    div WORD [sectors_per_track] ; AX = Cylinder, DX = Sector (1-based)

    ; Load CHS values into the correct registers
    mov ch, al              ; Cylinder (lower byte into CH)
    xor dh, dh              ; DH = Head (0 for CD-ROM)
    inc dl                  ; Convert DX (sector) to 1-based
    mov cl, dl              ; CL = Sector number

    ; BIOS INT 13h - Read sector
    mov ah, 0x02            ; AH = 0x02 (Read Sector)
    mov al, 0x01            ; AL = Number of sectors to read (1 sector = 512 bytes)
    int 0x13                ; Call BIOS Disk Service to read sector

    jc read_error           ; If error (Carry Flag set), jump to error handler

    ; Sector read successfully
    pop dx
    pop cx
    pop bx
    pop ax
    ret                     ; Return to caller

read_error:
    ; Print error message
    mov si, error_msg
    call print_string

    ; Output error code (in AH) as a hexadecimal value
    mov ah, 0x00            ; Reset AH to clear previous content
    mov al, ah              ; Move error code from AH to AL
    call print_hex           ; Print the error code in hexadecimal format

    jmp hang

; Function: print_hex
; Prints the value in AL as a 2-digit hexadecimal number
print_hex:
    push ax
    push bx

    ; Print high nibble
    mov bl, al
    shr al, 4               ; Shift the high nibble into the lower 4 bits
    call print_nibble

    ; Print low nibble
    mov al, bl
    and al, 0x0F            ; Mask out the upper nibble
    call print_nibble

    pop bx
    pop ax
    ret

; Function: print_nibble
; Prints the nibble (low 4 bits) of AL as a hexadecimal digit
print_nibble:
    add al, '0'             ; Convert nibble to ASCII
    cmp al, '9'             ; If it's greater than '9', adjust for letters
    jle .print_char
    add al, 7               ; Adjust ASCII for 'A' through 'F'

.print_char:
    mov ah, 0x0E            ; BIOS teletype output
    int 0x10                ; Print the character in AL
    ret


; Function: print_string
; Print a null-terminated string pointed to by SI to the screen
print_string:
    mov ah, 0x0E           ; BIOS teletype output
.print_char:
    lodsb                  ; Load byte at [SI] into AL
    cmp al, 0               ; Check if it's the null terminator
    je .done
    int 0x10               ; BIOS interrupt to print character in AL
    jmp .print_char
	
.done:
    ret

boot_msg db 'Booting kernel from CD-ROM...', 0
error_msg db 'Error reading from CD-ROM!', 0

sectors_per_track dw 63    ; Common value for sectors per track

TIMES 510-($-$$) db 0      ; Pad boot sector with zeros up to 510 bytes
DW 0xAA55                  ; Boot signature (0x55AA)
