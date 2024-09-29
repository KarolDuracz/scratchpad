[bits 16]              ; Real mode (16-bit code)
[org 0x7C00]           ; BIOS loads bootloader here in real mode

start:
    ; Clear registers
    xor ax, ax
    xor bx, bx
    xor cx, cx
    xor dx, dx

    ; Display message (optional)
    mov si, boot_message
    call print_string

    ; Prepare disk address packet for INT 13h, AH = 42h (BIOS Extended Read)
    mov ax, 0x07C0      ; Load to 0x07C0:0000 (right after bootloader in memory)
    mov es, ax
    mov bx, 0x0000      ; Buffer offset

    ; Prepare disk address packet
    mov byte [DAP], 0x10       ; Size of DAP (16 bytes)
    mov byte [DAP+1], 0x00     ; Reserved (0)
    mov word [DAP+2], 1        ; Number of sectors to read
    mov word [DAP+4], 0x0000   ; Buffer offset in memory (0000)
    mov word [DAP+6], 0x07C0   ; Buffer segment in memory (07C0)
    mov dword [DAP+8], 0x10    ; Starting LBA (Logical Block Address)

    ; Call BIOS to read sector from CD-ROM
    mov ah, 0x42        ; BIOS Extended Read
    mov dl, 0xE0        ; CD-ROM drive number (0xE0 for first CD-ROM)
    mov si, DAP         ; Pointer to Disk Address Packet (DAP)
    int 0x13            ; Call BIOS

    jc error            ; Jump to error if carry flag is set

    ; Continue loading the bootloader from CD (if successful)
    ; Jump to the loaded code (e.g., at 0x07C0:0000)
    jmp 0x07C0:0000

error:
    ; Handle read error (optional)
    mov si, error_message
    call print_string
    cli
    hlt

; Print a string (terminated by '$')
print_string:
    mov ah, 0x0E
.next_char:
    lodsb
    cmp al, '$'
    je .done
    int 0x10
    jmp .next_char
.done:
    ret

; Data
boot_message db "Booting from CD-ROM...$", 0
error_message db "Error reading CD-ROM!$", 0

; Disk Address Packet (DAP) structure
DAP:
    db 0x10, 0x00       ; DAP size and reserved byte
    dw 1                ; Number of sectors to read
    dw 0x0000           ; Buffer offset
    dw 0x07C0           ; Buffer segment
    dq 0x10             ; Starting LBA (Logical Block Address)

times 510-($-$$) db 0   ; Padding to 510 bytes
dw 0xAA55               ; Boot signature (0xAA55)
