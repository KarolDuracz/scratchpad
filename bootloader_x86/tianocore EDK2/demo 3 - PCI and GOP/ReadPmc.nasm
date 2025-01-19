;------------------------------------------------------------------------------
;
; Copyright (c) 2006, Intel Corporation. All rights reserved.<BR>
; SPDX-License-Identifier: BSD-2-Clause-Patent
;
; Module Name:
;
;   ReadPmc.Asm
;
; Abstract:
;
;   AsmReadPmc function
;
; Notes:
;
;------------------------------------------------------------------------------

    DEFAULT REL

	
	SECTION .data
		;global ASM_PFX(_vartest)
		extern _vartest
		
		; this is for calculation ACNT/MCNT
		extern _vartest2
		extern _vartest3
	
    SECTION .text

;------------------------------------------------------------------------------
; UINT64
; EFIAPI
; AsmReadPmc (
;   IN UINT32   PmcIndex
;   );
;------------------------------------------------------------------------------
;global ASM_PFX(AsmReadPmc)
;ASM_PFX(AsmReadPmc):
;    rdpmc
;    shl     rdx, 0x20
;    or      rax, rdx
;    ret

global ASM_PFX(AsmReadPmc)
ASM_PFX(AsmReadPmc):
	;mov eax, 0xabcd
    ;mov [_vartest], eax
    
	cmp rcx, 0xff
	je label1
	cmp rcx, 0xa
	je label2
	cmp rcx, 0x20
	je label3
	cmp rcx, 0x40
	je label4
	cmp rcx, 0x50
	je label5
	cmp rcx, 0x60
	je label6
	cmp rcx, 0x70
	je label7
	cmp rcx, 0x80
	je label8
	
	; 64 bit test for print as hex
	cmp rcx, 0x90
	je label9
	cmp rcx, 0x100
	je label10
	cmp rcx, 0x110
	je label11
	
	; MSR_RAPL_POWER_UNIT
	cmp rcx, 0x120
	je label12
	
	cmp rcx, 0x130 ; MSR_PLTFORM_INFO 0xce
	je label13
	cmp rcx, 0x140 ; MSR_PERF_STATUS 0x198
	je label14
	cmp rcx, 0x150 ; MSR_PERF_CTL 0x199 - nastepne po 0x198 
	je label15
	cmp rcx, 0x160 ; MSR_TURBO_RATIO_LIMIT
	je label16
	
	; first test for 64 bit value
	cmp rcx, 0x200
	je label200
	
	
	
	mov rax, 0xabcd
	mov [_vartest], rax

label1:
	mov rax, 0x1
	mov [_vartest], rax
	ret
	
label2:
	mov rax, rcx
	mov [_vartest], rax
	ret
	
label3:
	;mov rcx, 0x1b
	;rdmsr
	
	mov rax, cs
	mov [_vartest], rax
	ret

label4:
	cli
	mov rax, cr4
	;mov rdx, 1
	mov [_vartest], rax
	sti
	ret
	
label5:
	cli 
	mov rcx, 0x1b
	rdmsr
	mov [_vartest], rax
	sti
	ret
	
label6:
	cli 
	mov rax, cr0
	mov [_vartest], rax
	sti
	ret
	
label7:
	cli 
	mov rcx, 0xe7
	rdmsr
	mov [_vartest], rax
	sti
	ret

label8:
	cli 
	mov rcx, 0xe8
	rdmsr
	mov [_vartest], rax
	sti
	ret

; label 9 oraz 10 to wersje pod lable 11 ktora robi DIV
; domyslnie do oczytywania rdmsr uzywaj (kopiuj) label 7 oraz lebel 8

label9:
	cli 
	mov rcx, 0xe8
	rdmsr
	mov [_vartest], rax ; to jest potrzebne dla label 11 dla DIV
	mov [_vartest2], rax ; to jest potrzebne dla label 11 dla DIV
	sti
	ret
	
label10:
	cli 
	mov rcx, 0xe7
	rdmsr
	mov [_vartest], rax ; to jest potrzebne dla label 11 dla DIV
	mov [_vartest3], rax ; to jest potrzebne dla label 11 dla DIV
	sti
	ret

label11:	
	push rbx
	push rdx
	push rcx
	mov rcx, [_vartest2]
	mov rax, [_vartest3]
	div rcx
	mov [_vartest], rax
	pop rcx
	pop rdx
	pop rbx
	ret

label12:
	cli 
	mov rcx, 0x606
	rdmsr
	mov [_vartest], rax
	sti
	ret

; MSR_PLATFORM_INFO
label13:
	cli 
	mov rcx, 0xce
	rdmsr
	mov [_vartest], rax
	sti
	ret

; MSR PERF STATUS
label14:
	cli 
	mov rcx, 0x198
	rdmsr
	mov [_vartest], rax
	sti
	ret

; IA32_PERF_CTL
label15:
	cli 
	mov rcx, 0x199
	rdmsr
	mov [_vartest], rax
	sti
	ret
	
; MSR_TURBO_RATIO_LIMIT
label16:
	cli 
	mov rcx, 0x1ad
	rdmsr
	mov [_vartest], rax
	sti
	ret

label200:
	mov rax, 0x1234567890abcdef
	mov [_vartest], rax
	ret
	