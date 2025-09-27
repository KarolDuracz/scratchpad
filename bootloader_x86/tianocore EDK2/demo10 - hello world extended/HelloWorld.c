/** @file
  This sample application bases on HelloWorld PCD setting
  to print "UEFI Hello World!" to the UEFI Console.

  Copyright (c) 2006 - 2018, Intel Corporation. All rights reserved.<BR>
  SPDX-License-Identifier: BSD-2-Clause-Patent

**/

/*
#include <Uefi.h>
#include <Library/PcdLib.h>
#include <Library/UefiLib.h>
#include <Library/UefiApplicationEntryPoint.h>
*/

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
/*
EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  UINT32  Index;

  Index = 0;

  //
  // Three PCD type (FeatureFlag, UINT32 and String) are used as the sample.
  //
  if (FeaturePcdGet (PcdHelloWorldPrintEnable)) {
    for (Index = 0; Index < PcdGet32 (PcdHelloWorldPrintTimes); Index++) {
      //
      // Use UefiLib Print API to print string to UEFI console
      //
      Print ((CHAR16 *)PcdGetPtr (PcdHelloWorldPrintString));
    }
  }

  return EFI_SUCCESS;
}
*/

/** HelloBootServicesTest.c
  Simple UEFI application that tests EFI_SYSTEM_TABLE and EFI_BOOT_SERVICES
  operations: prints pointers, allocates/free pool, locates Timer Arch Protocol,
  queries timer period, creates a timer event, sets a periodic timer and waits
  for a few ticks, then cleans up.

  Build as a UEFI_APPLICATION (HelloBootServicesTest.inf).
**/


// TEST 1 - OK
/*
#include <Uefi.h>
#include <Library/UefiLib.h>                // Print()
#include <Library/UefiApplicationEntryPoint.h>
#include <Library/MemoryAllocationLib.h>    // AllocatePool, FreePool
#include <Library/BaseMemoryLib.h>          // SetMem
#include <Library/PrintLib.h>               // AsciiSPrint if desired
#include <Protocol/Timer.h>                 // gEfiTimerArchProtocolGuid
#include <Library/UefiBootServicesTableLib.h>

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  //EFI_BOOT_SERVICES *gBS = SystemTable->BootServices;
  EFI_STATUS Status;
  VOID *Buffer = NULL;
  UINTN BufSize = 128;
  EFI_EVENT TimerEvent = NULL;
  UINTN Index;
  UINTN WaitIndex;
  EFI_TIMER_ARCH_PROTOCOL *TimerArch = NULL;
  UINT64 Period100ns = 0;

  Print (L"\n=== HelloBootServicesTest: start ===\n");

  // Print SystemTable and BootServices pointers
  Print (L"SystemTable: %p\n", SystemTable);
  Print (L"ConOut:      %p\n", SystemTable->ConOut);
  Print (L"BootServices:%p\n\n", gBS);

  // AllocatePool test
  Status = gBS->AllocatePool(EfiBootServicesData, BufSize, &Buffer);
  Print (L"AllocatePool(%u) -> %r\n", BufSize, Status);
  if (!EFI_ERROR (Status) && Buffer != NULL) {
    SetMem(Buffer, BufSize, 0xA5);
    Print (L"Buffer[0]=0x%02x BufAddr=%p\n", ((UINT8*)Buffer)[0], Buffer);
    Status = gBS->FreePool(Buffer);
    Print (L"FreePool -> %r\n", Status);
    Buffer = NULL;
  }

  // Locate Timer Arch Protocol (optional: might not exist on some targets)
  Status = gBS->LocateProtocol(&gEfiTimerArchProtocolGuid, NULL, (VOID**)&TimerArch);
  Print (L"LocateProtocol(TimerArch) -> %r\n", Status);
  if (!EFI_ERROR(Status) && TimerArch != NULL) {
    Status = TimerArch->GetTimerPeriod(TimerArch, &Period100ns);
    Print (L"TimerArch->GetTimerPeriod -> %r (value=%llu [100ns units])\n", Status, Period100ns);
    // Convert to ns for display
    if (!EFI_ERROR(Status)) {
      UINT64 PeriodNs = MultU64x32(Period100ns, 100);
      Print (L"Timer period: %llu ns (~%llu us)\n", PeriodNs, DivU64x32(PeriodNs, 1000));
    }
  }

  //
  // Create a timer event that we can WaitForEvent() on.
  // Use EVT_TIMER (no notify) and call SetTimer to make it periodic.
  //
  Status = gBS->CreateEvent(EVT_TIMER, TPL_APPLICATION, NULL, NULL, &TimerEvent);
  Print (L"CreateEvent(EVT_TIMER) -> %r (Event=%p)\n", Status, TimerEvent);

  if (!EFI_ERROR(Status) && TimerEvent != NULL) {
    // Set periodic timer: argument is in 100ns units. 10,000,000 == 1 second.
    // TimerPeriodic is standard UEFI enum.
    UINT64 Period100nsForOneSecond = 10000000ULL; // 1 s
    Status = gBS->SetTimer(TimerEvent, TimerPeriodic, Period100nsForOneSecond);
    Print (L"SetTimer(periodic, 1s) -> %r\n", Status);

    if (!EFI_ERROR(Status)) {
      // Wait for N ticks (demonstrates WaitForEvent)
      const UINTN TicksToWait = 3;
      for (Index = 0; Index < TicksToWait; Index++) {
        Status = gBS->WaitForEvent(1, &TimerEvent, &WaitIndex);
        Print (L"WaitForEvent returned: %r (index=%u)  tick=%u\n", Status, WaitIndex, (UINT32)(Index+1));
        if (EFI_ERROR(Status)) break;
      }
      // Cancel timer: SetTimer with TimerCancel or set relative 0 — many implementations
      // simply close the event to stop the timer. Use SetTimer(..., 0) to try cancel if supported.
      Status = gBS->SetTimer(TimerEvent, TimerPeriodic, 0);
      Print (L"SetTimer(cancel) -> %r\n", Status);
    }

    // Close the event
    Status = gBS->CloseEvent(TimerEvent);
    Print (L"CloseEvent -> %r\n", Status);
    TimerEvent = NULL;
  }

  Print (L"=== HelloBootServicesTest: end ===\n\n");
  return EFI_SUCCESS;
}
*/


// TEST 2 - ? - OKOOOKKK

/** HelloBootServicesTest.c
  Simple UEFI application that tests EFI_SYSTEM_TABLE and EFI_BOOT_SERVICES
  operations including InstallProtocolInterface/LocateProtocol/UninstallProtocolInterface.
**/

#include <Uefi.h>
#include <Library/UefiLib.h>                // Print()
#include <Library/UefiApplicationEntryPoint.h>
#include <Library/MemoryAllocationLib.h>    // AllocatePool, FreePool
#include <Library/BaseMemoryLib.h>          // SetMem
#include <Library/PrintLib.h>               // AsciiSPrint if desired
#include <Protocol/Timer.h>                 // gEfiTimerArchProtocolGuid
#include <Library/UefiBootServicesTableLib.h>

//
// If the protocol GUID declaration header exists in your tree, you can include it,
// but it's not mandatory for the test because we declare the GUID in the INF [Protocols].
// #include <Protocol/GenericMemoryTest.h>
//

EFI_STATUS
EFIAPI
UefiMain2 (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  //EFI_BOOT_SERVICES *gBS = SystemTable->BootServices;
  EFI_STATUS Status;
  VOID *Buffer = NULL;
  UINTN BufSize = 128;
  EFI_EVENT TimerEvent = NULL;
  UINTN Index;
  UINTN WaitIndex;
  EFI_TIMER_ARCH_PROTOCOL *TimerArch = NULL;
  UINT64 Period100ns = 0;

  // Variables used for protocol install test
  EFI_HANDLE mGenericMemoryTestHandle = NULL;
  UINT8 mGenericMemoryTestInterface = 0xCC; // placeholder interface object
  VOID *LocatedInterface = NULL;

  Print (L"\n=== HelloBootServicesTest: start ===\n");

  // Print pointers
  Print (L"SystemTable: %p\n", SystemTable);
  Print (L"ConOut:      %p\n", SystemTable->ConOut);
  Print (L"BootServices:%p\n\n", gBS);

  // AllocatePool test
  Status = gBS->AllocatePool(EfiBootServicesData, BufSize, &Buffer);
  Print (L"AllocatePool(%u) -> %r\n", BufSize, Status);
  if (!EFI_ERROR (Status) && Buffer != NULL) {
    SetMem(Buffer, BufSize, 0xA5);
    Print (L"Buffer[0]=0x%02x BufAddr=%p\n", ((UINT8*)Buffer)[0], Buffer);
    Status = gBS->FreePool(Buffer);
    Print (L"FreePool -> %r\n", Status);
    Buffer = NULL;
  }

  // Locate Timer Arch Protocol (optional)
  Status = gBS->LocateProtocol(&gEfiTimerArchProtocolGuid, NULL, (VOID**)&TimerArch);
  Print (L"LocateProtocol(TimerArch) -> %r\n", Status);
  if (!EFI_ERROR(Status) && TimerArch != NULL) {
    Status = TimerArch->GetTimerPeriod(TimerArch, &Period100ns);
    Print (L"TimerArch->GetTimerPeriod -> %r (value=%llu [100ns units])\n", Status, Period100ns);
    if (!EFI_ERROR(Status)) {
      UINT64 PeriodNs = MultU64x32(Period100ns, 100);
      Print (L"Timer period: %llu ns (~%llu us)\n", PeriodNs, DivU64x32(PeriodNs, 1000));
    }
  }

  //
  // Create a timer event that we can WaitForEvent() on.
  //
  Status = gBS->CreateEvent(EVT_TIMER, TPL_APPLICATION, NULL, NULL, &TimerEvent);
  Print (L"CreateEvent(EVT_TIMER) -> %r (Event=%p)\n", Status, TimerEvent);

  if (!EFI_ERROR(Status) && TimerEvent != NULL) {
    UINT64 Period100nsForOneSecond = 10000000ULL; // 1 s
    Status = gBS->SetTimer(TimerEvent, TimerPeriodic, Period100nsForOneSecond);
    Print (L"SetTimer(periodic, 1s) -> %r\n", Status);

    if (!EFI_ERROR(Status)) {
      const UINTN TicksToWait = 2;
      for (Index = 0; Index < TicksToWait; Index++) {
        Status = gBS->WaitForEvent(1, &TimerEvent, &WaitIndex);
        Print (L"WaitForEvent returned: %r (index=%u)  tick=%u\n", Status, WaitIndex, (UINT32)(Index+1));
        if (EFI_ERROR(Status)) break;
      }
      // Cancel timer by setting period 0 (attempt)
      Status = gBS->SetTimer(TimerEvent, TimerPeriodic, 0);
      Print (L"SetTimer(cancel) -> %r\n", Status);
    }

    Status = gBS->CloseEvent(TimerEvent);
    Print (L"CloseEvent -> %r\n", Status);
    TimerEvent = NULL;
  }

  //
  // ---- START: InstallProtocolInterface test using gEfiGenericMemTestProtocolGuid ----
  //
  Print (L"\n-- InstallProtocolInterface test start --\n");

  Status = gBS->InstallProtocolInterface (
                  &mGenericMemoryTestHandle,
                  &gEfiGenericMemTestProtocolGuid,
                  EFI_NATIVE_INTERFACE,
                  &mGenericMemoryTestInterface
                  );
  Print (L"InstallProtocolInterface -> %r (Handle=%p)\n", Status, mGenericMemoryTestHandle);

  if (!EFI_ERROR(Status)) {
    // Try to locate the protocol we just installed
    Status = gBS->LocateProtocol(&gEfiGenericMemTestProtocolGuid, NULL, &LocatedInterface);
    Print (L"LocateProtocol(gEfiGenericMemTestProtocolGuid) -> %r Located=%p\n", Status, LocatedInterface);

    // Check that the located interface points to our object (it should)
    if (!EFI_ERROR(Status) && LocatedInterface == &mGenericMemoryTestInterface) {
      Print (L"Located interface matches installed pointer (ok)\n");
    } else {
      Print (L"Located interface does NOT match installed pointer (found different object)\n");
    }

    // Now uninstall the protocol
    Status = gBS->UninstallProtocolInterface(mGenericMemoryTestHandle, &gEfiGenericMemTestProtocolGuid, &mGenericMemoryTestInterface);
    Print (L"UninstallProtocolInterface -> %r\n", Status);

    // Optionally free or close handle if necessary (handle can remain for other installs)
    // In many simple cases you can leave handle; but if you allocated resources tied to it clean them up.
    mGenericMemoryTestHandle = NULL;
  } else {
    Print (L"InstallProtocolInterface failed; skipping locate/uninstall\n");
  }

  Print (L"-- InstallProtocolInterface test end --\n\n");

  Print (L"=== HelloBootServicesTest: end ===\n\n");
  return EFI_SUCCESS;
}
