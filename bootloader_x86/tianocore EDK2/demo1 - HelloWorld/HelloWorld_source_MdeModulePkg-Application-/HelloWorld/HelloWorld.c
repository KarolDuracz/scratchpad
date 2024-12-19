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

  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_YELLOW, EFI_GREEN));
  SystemTable->ConOut->ClearScreen(SystemTable->ConOut);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"Hello, World!\r\n\r\n");
  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_RED, EFI_BLACK));
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"Press any key to shutdown");
  
  UINT32 eax, ebx, ecx, edx;
  AsmCpuid(0x06, &eax, &ebx, &ecx, &edx);
  
  
  
  CHAR16 buffer[64];
  //UnicodeSPrint(buffer, sizeof(buffer), L"eax: 0x%08x\n", eax);
  //UnicodeValueToString(buffer, LEFT_JUSTIFY, eax, 16);
  
  CHAR16 *hexchar = L"01234567890ABCDEF";
  
  for (INTN i = 0; i < 8; i++) {
	buffer[7 - i] = hexchar[(eax >> (i * 4)) & 0xf];
  }
  buffer[8] = L'\0';
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  
  EFI_INPUT_KEY key;
  while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS);
  SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
  
  
  //Print(L"Hi\n");
  
  return EFI_SUCCESS;
}
