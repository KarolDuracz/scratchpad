/** @file
  This sample application bases on HelloWorld PCD setting
  to print "UEFI Hello World!" to the UEFI Console.

  Copyright (c) 2006 - 2018, Intel Corporation. All rights reserved.<BR>
  SPDX-License-Identifier: BSD-2-Clause-Patent

**/

#include <Uefi.h>
#include <Library/PcdLib.h>
#include <Library/UefiLib.h>
#include <Library/BaseLib.h>
#include <Library/PrintLib.h>
#include <Library/UefiApplicationEntryPoint.h>

extern UINT64 _vartest;
extern UINT64 _vartest2;
extern UINT64 _vartest3;
extern UINT64 _vartest4_timer1;

//
// String token ID of help message text.
// Shell supports to find help message in the resource section of an application image if
// .MAN file is not found. This global variable is added to make build tool recognizes
// that the help string is consumed by user and then build tool will add the string into
// the resource section. Thus the application can use '-?' option to show help message in
// Shell.
//
GLOBAL_REMOVE_IF_UNREFERENCED EFI_STRING_ID  mStringHelpTokenId = STRING_TOKEN (STR_HELLO_WORLD_HELP_INFORMATION);

/**
  The user Entry Point for Application. The user code starts with this function
  as the real entry point for the application.

  @param[in] ImageHandle    The firmware allocated handle for the EFI image.
  @param[in] SystemTable    A pointer to the EFI System Table.

  @retval EFI_SUCCESS       The entry point is executed successfully.
  @retval other             Some error occurs when executing this entry point.

**/
EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
	
	
	//CpuBreakpoint();
	//DEBUG ((DEBUG_INFO, " [ BIOS test ] demo1 1 %d\n", ImageHande));
	//CHAR8 buf1[64];
	////ValueToHexStr(buf1, ImageHandle, NULL, sizoef(ImageHandle));

  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_YELLOW, EFI_GREEN));
  SystemTable->ConOut->ClearScreen(SystemTable->ConOut);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"Hello, World!\r\n\r\n");
  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_RED, EFI_BLACK));
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"Press any key to shutdown");
  
  UINT32 eax, ebx, ecx, edx;
  //AsmCpuid(0x06, &eax, &ebx, &ecx, &edx);
  
  AsmCpuid(0x01, &eax, &ebx, &ecx, &edx);
  
  CHAR16 buffer[64]; // eax
  CHAR16 ebxb[64];
  CHAR16 ecxb[64];
  CHAR16 edxb[64];
  // espb[64];
  //CHAR16 rbpb[64];
  //UnicodeSPrint(buffer, sizeof(buffer), L"eax: 0x%08x\n", eax);
  //UnicodeValueToString(buffer, LEFT_JUSTIFY, eax, 16);
  
  CHAR16 *hexchar = L"0123456789ABCDEF";
  
  for (INTN i = 0; i < 8; i++) {
	buffer[7 - i] = hexchar[(eax >> (i * 4)) & 0xf];
	ebxb[7 - i] = hexchar[(ebx >> (i * 4)) & 0xf];
	ecxb[7 - i] = hexchar[(ecx >> (i * 4)) & 0xf];
	edxb[7 - i] = hexchar[(edx >> (i * 4)) & 0xf];
  }
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0';
  
  ebxb[8] = L'\r';
  ebxb[9] = L'\n';
  ebxb[10] = L'\0';
  
  ecxb[8] = L'\r';
  ecxb[9] = L'\n';
  ecxb[10] = L'\0';
  
  edxb[8] = L'\r';
  edxb[9] = L'\n';
  edxb[10] = L'\0';
  
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, ebxb);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, ecxb);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, edxb);
  
  //custom_code();
  
  //UINT32 lowB;
  //UINT32 highB;
  //asm ("rtdsc" : "=a"(lowB), "=d" (highB));
  
 
  
  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_YELLOW, EFI_GREEN));
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"RTDSC!\r\n\r\n");
  
  
  //AsmInvd(0x0, &eax, &ebx, &ecx, &edx);
  _vartest = 0x1234;
  AsmInvd();
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"read cr0!\r\n\r\n");
  
  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	//ebxb[7 - j] = hexchar[(ebx >> (j * 4)) & 0xf];
	//ecxb[7 - j] = hexchar[(ecx >> (j * 4)) & 0xf];
	//edxb[7 - j] = hexchar[(edx >> (j * 4)) & 0xf];
  }
  //buffer[8] = L'\0';
  
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0'; 
  
  /*
  ebxb[8] = L'\r';
  ebxb[9] = L'\n';
  ebxb[10] = L'\0';
  
  ecxb[8] = L'\r';
  ecxb[9] = L'\n';
  ecxb[10] = L'\0';
  
  edxb[8] = L'\r';
  edxb[9] = L'\n';
  edxb[10] = L'\0';
  */
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"DONE!\r\n\r\n");
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, ebxb);
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, ecxb);
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, edxb);
  
  // 05-01-2025 - 22:04 
  // ok, skoro to dziala już i mogę odtyczyća _vartest jako extern z .nasm
  // to teraz test RDTSC
  
  _vartest = 0;
  
  AsmInvd();

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	//ebxb[7 - j] = hexchar[(ebx >> (j * 4)) & 0xf];
	//ecxb[7 - j] = hexchar[(ecx >> (j * 4)) & 0xf];
	//edxb[7 - j] = hexchar[(edx >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  // demo 2 - ustawiam _vartest = 1
  CHAR16 buffer_var2[64];
  CHAR16 buffer_var3[64];
  _vartest = 1;
  
  AsmInvd();

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	buffer_var2[7 - j] = hexchar[(_vartest2 >> (j * 4)) & 0xf];
	buffer_var3[7 - j] = hexchar[(_vartest3 >> (j * 4)) & 0xf];
	//ebxb[7 - j] = hexchar[(ebx >> (j * 4)) & 0xf];
	//ecxb[7 - j] = hexchar[(ecx >> (j * 4)) & 0xf];
	//edxb[7 - j] = hexchar[(edx >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0';
  
  buffer_var2[8] = L'\r';
  buffer_var2[9] = L'\n';
  buffer_var2[10] = L'\0';
  
  buffer_var3[8] = L'\r';
  buffer_var3[9] = L'\n';
  buffer_var3[10] = L'\0';
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_var2);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_var3);
  
  // next demo
  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_RED, EFI_BLACK));
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, u"\r\n\DEMO 3!\r\n");
  
  // wait for  key event 
  EFI_INPUT_KEY key;
  while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS) {
	//SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_RED, EFI_BLACK));
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 10);
	SystemTable->ConOut->OutputString(SystemTable->ConOut, u"\rDEMO 3!");
	
	for (INTN j = 0; j < 8; j++) {
		buffer[7 - j] = hexchar[(_vartest4_timer1 >> (j * 4)) & 0xf];
	}
	  buffer[8] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
		
	// inc timer
	_vartest4_timer1++;
  }
  
  // shutdown procedure
  SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
  
  
  //Print(L"Hi\n");
  
  return EFI_SUCCESS;
}
