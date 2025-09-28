/** HelloBootServicesTest.c
  Consolidated and refactored test application for Boot Services + ACPI tests.

  - Installs a small ACPI protocol stub if none is present.
  - Writes a minimal ACPI RSDP by offsets into an allocated buffer and installs it
    into the System Configuration Table (gEfiAcpi20TableGuid) so shell "dmem"
    and other tools detect ACPI.
  - Attempts a safe InstallAcpiTable/UninstallAcpiTable test.
**/

#include <Uefi.h>

#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h> // gBS
#include <Library/MemoryAllocationLib.h>      // AllocatePool / FreePool
#include <Library/BaseMemoryLib.h>            // CopyMem, SetMem, CompareMem
#include <Library/PrintLib.h>                 // Print helpers
#include <Library/BaseLib.h>                  // MultU64x32, DivU64x32 (if needed)
#include <Library/DebugLib.h>

#include <Protocol/Timer.h>
#include <Protocol/AcpiTable.h>
#include <Guid/Acpi.h>
#include <IndustryStandard/Acpi.h>            // optional, used for EFI_ACPI_DESCRIPTION_HEADER

// Minimal linked-list node for stubbed tables
typedef struct ACPI_STUB_NODE_ {
  UINTN                     Key;
  VOID                     *Table;
  UINTN                     Size;
  struct ACPI_STUB_NODE_   *Next;
} ACPI_STUB_NODE;

STATIC ACPI_STUB_NODE *mAcpiStubList = NULL;
STATIC UINTN          mAcpiNextKey  = 1;
STATIC EFI_HANDLE     mAcpiStubHandle = NULL; // handle for installed stub protocol

// Forward declarations
STATIC
EFI_STATUS
EFIAPI
StubInstallAcpiTable (
  IN EFI_ACPI_TABLE_PROTOCOL  *This,
  IN VOID                     *Table,
  IN UINTN                    TableSize,
  OUT UINTN                   *TableKey
  );

STATIC
EFI_STATUS
EFIAPI
StubUninstallAcpiTable (
  IN EFI_ACPI_TABLE_PROTOCOL  *This,
  IN UINTN                    TableKey
  );

// Stub protocol instance (function pointers)
STATIC EFI_ACPI_TABLE_PROTOCOL mStubAcpiProtocol = {
  StubInstallAcpiTable,
  StubUninstallAcpiTable
};

/**
  Minimal InstallAcpiTable: copy the table into allocated pool and return a key.
**/
STATIC
EFI_STATUS
EFIAPI
StubInstallAcpiTable (
  IN EFI_ACPI_TABLE_PROTOCOL  *This,
  IN VOID                     *Table,
  IN UINTN                    TableSize,
  OUT UINTN                   *TableKey
  )
{
  ACPI_STUB_NODE *Node;
  VOID           *Copy;
  EFI_STATUS      Status;

  if (Table == NULL || TableSize == 0 || TableKey == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  Status = gBS->AllocatePool (EfiBootServicesData, TableSize, &Copy);
  if (EFI_ERROR (Status)) {
    return Status;
  }

  // copy table contents
  CopyMem ((VOID *) Copy, (CONST VOID *) Table, TableSize);

  Status = gBS->AllocatePool (EfiBootServicesData, sizeof (ACPI_STUB_NODE), (VOID **)&Node);
  if (EFI_ERROR (Status)) {
    gBS->FreePool (Copy);
    return Status;
  }

  Node->Key   = mAcpiNextKey++;
  Node->Table = Copy;
  Node->Size  = TableSize;
  Node->Next  = mAcpiStubList;
  mAcpiStubList = Node;

  *TableKey = Node->Key;

  return EFI_SUCCESS;
}

/**
  Minimal UninstallAcpiTable: find by key, free stored table and node.
**/
STATIC
EFI_STATUS
EFIAPI
StubUninstallAcpiTable (
  IN EFI_ACPI_TABLE_PROTOCOL  *This,
  IN UINTN                    TableKey
  )
{
  ACPI_STUB_NODE *Prev = NULL;
  ACPI_STUB_NODE *Node = mAcpiStubList;

  while (Node != NULL) {
    if (Node->Key == TableKey) {
      break;
    }
    Prev = Node;
    Node = Node->Next;
  }

  if (Node == NULL) {
    return EFI_NOT_FOUND;
  }

  if (Prev == NULL) {
    mAcpiStubList = Node->Next;
  } else {
    Prev->Next = Node->Next;
  }

  if (Node->Table != NULL) {
    gBS->FreePool (Node->Table);
  }
  gBS->FreePool (Node);

  return EFI_SUCCESS;
}

/**
  Ensure an ACPI Table Protocol is available. If none present, install local stub.
  Returns EFI_SUCCESS and sets OutProtocol to a usable pointer (real or stub).
**/
STATIC
EFI_STATUS
EnsureAcpiTableProtocolAvailable (
  OUT EFI_ACPI_TABLE_PROTOCOL **OutProtocol
  )
{
  EFI_STATUS                Status;
  EFI_ACPI_TABLE_PROTOCOL  *Protocol = NULL;

  if (OutProtocol == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  Status = gBS->LocateProtocol (&gEfiAcpiTableProtocolGuid, NULL, (VOID **)&Protocol);
  if (!EFI_ERROR (Status) && Protocol != NULL) {
    *OutProtocol = Protocol;
    return EFI_SUCCESS;
  }

  // Install stub protocol
  Status = gBS->InstallProtocolInterface (
                  &mAcpiStubHandle,
                  &gEfiAcpiTableProtocolGuid,
                  EFI_NATIVE_INTERFACE,
                  &mStubAcpiProtocol
                  );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  // Re-locate to return the protocol pointer
  Status = gBS->LocateProtocol (&gEfiAcpiTableProtocolGuid, NULL, (VOID **)&Protocol);
  if (EFI_ERROR (Status) || Protocol == NULL) {
    return EFI_NOT_FOUND;
  }

  *OutProtocol = Protocol;
  return EFI_SUCCESS;
}

/**
  Fill a raw buffer with a minimal ACPI 2.0 RSDP by offsets (no struct field access).
  Buffer must be at least 36 bytes (ACPI 2.0 RSDP length).
**/
STATIC
EFI_STATUS
FillRsdpByOffsets (
  IN OUT VOID *RsdpBuf,
  IN     UINTN RsdpSize
  )
{
  enum {
    RSDP_SIG_OFF        = 0,   // 8 bytes
    RSDP_CHECKSUM_OFF   = 8,   // 1 byte
    RSDP_OEMID_OFF      = 9,   // 6 bytes
    RSDP_REVISION_OFF   = 15,  // 1 byte
    RSDP_RSDT_OFF       = 16,  // 4 bytes (UINT32 le)
    RSDP_LENGTH_OFF     = 20,  // 4 bytes (UINT32 le)
    RSDP_XSDT_OFF       = 24,  // 8 bytes (UINT64 le)
    RSDP_EXTCHK_OFF     = 32,  // 1 byte (extended checksum)
    RSDP_RESERVED_OFF   = 33,  // 3 bytes
    RSDP_MIN_SIZE       = 36   // ACPI 2.0 RSDP size
  };

  UINT8 *p;
  UINTN i;
  UINT8 sum8;

  if (RsdpBuf == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  if (RsdpSize < RSDP_MIN_SIZE) {
    return EFI_BAD_BUFFER_SIZE;
  }

  p = (UINT8 *) RsdpBuf;

  // Zero buffer for deterministic content
  SetMem ((VOID *) p, RsdpSize, 0);

  // Signature "RSD PTR " (8 bytes)
  CopyMem ((VOID *)&p[RSDP_SIG_OFF], (CONST VOID *)"RSD PTR ", 8);

  // OEMID (6 bytes)
  CopyMem ((VOID *)&p[RSDP_OEMID_OFF], (CONST VOID *)"EDK2EM", 6);

  // Revision = 2 (ACPI 2.0+)
  p[RSDP_REVISION_OFF] = 2;

  // RSDT/XSDT addresses = 0 (minimal test)
  for (i = 0; i < 4; ++i) { p[RSDP_RSDT_OFF + i] = 0; }

  // Length = RSDP_MIN_SIZE (4 bytes little-endian)
  {
    UINT32 len32 = (UINT32) RSDP_MIN_SIZE;
    p[RSDP_LENGTH_OFF + 0] = (UINT8) (len32 & 0xFF);
    p[RSDP_LENGTH_OFF + 1] = (UINT8) ((len32 >> 8) & 0xFF);
    p[RSDP_LENGTH_OFF + 2] = (UINT8) ((len32 >> 16) & 0xFF);
    p[RSDP_LENGTH_OFF + 3] = (UINT8) ((len32 >> 24) & 0xFF);
  }

  // XsdtAddress = 0 (8 bytes)
  for (i = 0; i < 8; ++i) { p[RSDP_XSDT_OFF + i] = 0; }

  // Clear checksum placeholders
  p[RSDP_CHECKSUM_OFF] = 0;
  p[RSDP_EXTCHK_OFF] = 0;

  // Compute legacy checksum (first 20 bytes)
  sum8 = 0;
  for (i = 0; i < 20; ++i) {
    sum8 = (UINT8) (sum8 + p[i]);
  }
  p[RSDP_CHECKSUM_OFF] = (UINT8) (0 - sum8);

  // Compute extended checksum over Length bytes
  {
    UINT32 length_le = (UINT32) (p[RSDP_LENGTH_OFF + 0]) |
                                 (p[RSDP_LENGTH_OFF + 1] << 8) |
                                 (p[RSDP_LENGTH_OFF + 2] << 16) |
                                 (p[RSDP_LENGTH_OFF + 3] << 24);

    if (length_le == 0 || length_le > RsdpSize) {
      // fallback to minimal
      length_le = (UINT32) RSDP_MIN_SIZE;
      p[RSDP_LENGTH_OFF + 0] = (UINT8) (length_le & 0xFF);
      p[RSDP_LENGTH_OFF + 1] = (UINT8) ((length_le >> 8) & 0xFF);
      p[RSDP_LENGTH_OFF + 2] = (UINT8) ((length_le >> 16) & 0xFF);
      p[RSDP_LENGTH_OFF + 3] = (UINT8) ((length_le >> 24) & 0xFF);
    }

    sum8 = 0;
    for (i = 0; i < (UINTN)length_le; ++i) {
      sum8 = (UINT8) (sum8 + p[i]);
    }
    p[RSDP_EXTCHK_OFF] = (UINT8) (0 - sum8);
  }

  return EFI_SUCCESS;
}

/**
  Allocate buffer, fill RSDP (via FillRsdpByOffsets), and install into System Configuration Table.
  On success returns pointer (in OutRsdp) which system keeps; caller must NOT free it on success.
**/
STATIC
EFI_STATUS
PrepareAndInstallMinimalRsdp (
  OUT VOID    **OutRsdp,
  IN  UINTN    RsdpSize,
  IN  EFI_SYSTEM_TABLE *SystemTable
  )
{
  EFI_STATUS Status;
  VOID *Buf = NULL;

  if (OutRsdp == NULL) {
    return EFI_INVALID_PARAMETER;
  }
  *OutRsdp = NULL;

  if (SystemTable == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  if (RsdpSize < 36) {
    return EFI_BAD_BUFFER_SIZE;
  }

  Status = gBS->AllocatePool (EfiBootServicesData, RsdpSize, &Buf);
  if (EFI_ERROR (Status) || Buf == NULL) {
    return Status;
  }

  // Fill by offsets
  Status = FillRsdpByOffsets (Buf, RsdpSize);
  if (EFI_ERROR (Status)) {
    gBS->FreePool (Buf);
    return Status;
  }

  // Try to install into System Configuration Table (ACPI 2.0 GUID)
  Status = gBS->InstallConfigurationTable (&gEfiAcpi20TableGuid, Buf);
  if (EFI_ERROR (Status)) {
    gBS->FreePool (Buf);
    return Status;
  }

  // Success: System Table now references Buf; caller should not free
  *OutRsdp = Buf;
  return EFI_SUCCESS;
}

/**
  Compute simple 8-bit checksum for buffer of length Len, return byte such that sum==0
**/
STATIC
UINT8
ComputeChecksum8 (
  IN VOID  *Buffer,
  IN UINTN Len
  )
{
  UINT8 *b = (UINT8 *) Buffer;
  UINT8 sum = 0;
  UINTN i;
  for (i = 0; i < Len; ++i) {
    sum = (UINT8)(sum + b[i]);
  }
  return (UINT8)(0 - sum);
}

/**
  Application entry point
**/
EFI_STATUS
EFIAPI
UefiMain3 (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;
  VOID *TempBuf = NULL;
  UINTN BufSize = 128;
  EFI_EVENT TimerEvent = NULL;
  UINTN Index;
  UINTN WaitIndex;
  EFI_TIMER_ARCH_PROTOCOL *TimerArch = NULL;
  UINT64 Period100ns = 0;

  Print (L"\n=== HelloBootServicesTest: start ===\n");

  // Print pointers
  Print (L"SystemTable: %p\n", SystemTable);
  Print (L"ConOut:      %p\n", SystemTable->ConOut);
  Print (L"BootServices:%p\n\n", gBS);

  // Small AllocatePool test
  Status = gBS->AllocatePool(EfiBootServicesData, BufSize, &TempBuf);
  Print (L"AllocatePool(%u) -> %r\n", (UINT32)BufSize, Status);
  if (!EFI_ERROR (Status) && TempBuf != NULL) {
    SetMem (TempBuf, BufSize, 0xA5);
    Print (L"Buffer[0]=0x%02x BufAddr=%p\n", ((UINT8*)TempBuf)[0], TempBuf);
    Status = gBS->FreePool(TempBuf);
    Print (L"FreePool -> %r\n", Status);
    TempBuf = NULL;
  }

  // Timer Arch Protocol test (optional)
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

  // Simple timer event test
  Status = gBS->CreateEvent(EVT_TIMER, TPL_APPLICATION, NULL, NULL, &TimerEvent);
  Print (L"CreateEvent(EVT_TIMER) -> %r (Event=%p)\n", Status, TimerEvent);
  if (!EFI_ERROR(Status) && TimerEvent != NULL) {
    Status = gBS->SetTimer(TimerEvent, TimerPeriodic, 10000000ULL); // 1s
    Print (L"SetTimer(periodic, 1s) -> %r\n", Status);
    if (!EFI_ERROR(Status)) {
      const UINTN TicksToWait = 2;
      for (Index = 0; Index < TicksToWait; Index++) {
        Status = gBS->WaitForEvent(1, &TimerEvent, &WaitIndex);
        Print (L"WaitForEvent returned: %r (index=%u)  tick=%u\n", Status, (UINT32)WaitIndex, (UINT32)(Index+1));
        if (EFI_ERROR(Status)) break;
      }
      Status = gBS->SetTimer(TimerEvent, TimerPeriodic, 0);
      Print (L"SetTimer(cancel) -> %r\n", Status);
    }
    Status = gBS->CloseEvent(TimerEvent);
    Print (L"CloseEvent -> %r\n", Status);
    TimerEvent = NULL;
  }

  //
  // InstallProtocolInterface test (generic memory-test proto)
  //
  {
    EFI_STATUS LocalStatus;
    EFI_HANDLE MemHandle = NULL;
    UINT8 MemIface = 0xCC;
    VOID *Located = NULL;

    Print (L"\n-- InstallProtocolInterface test start --\n");

    LocalStatus = gBS->InstallProtocolInterface (
                    &MemHandle,
                    &gEfiGenericMemTestProtocolGuid,
                    EFI_NATIVE_INTERFACE,
                    &MemIface
                    );
    Print (L"InstallProtocolInterface -> %r (Handle=%p)\n", LocalStatus, MemHandle);

    if (!EFI_ERROR(LocalStatus)) {
      LocalStatus = gBS->LocateProtocol(&gEfiGenericMemTestProtocolGuid, NULL, &Located);
      Print (L"LocateProtocol(gEfiGenericMemTestProtocolGuid) -> %r Located=%p\n", LocalStatus, Located);

      if (!EFI_ERROR(LocalStatus) && Located == &MemIface) {
        Print (L"Located interface matches installed pointer (ok)\n");
      } else {
        Print (L"Located interface does NOT match installed pointer (found different object)\n");
      }

      LocalStatus = gBS->UninstallProtocolInterface(MemHandle, &gEfiGenericMemTestProtocolGuid, &MemIface);
      Print (L"UninstallProtocolInterface -> %r\n", LocalStatus);
      MemHandle = NULL;
    } else {
      Print (L"InstallProtocolInterface failed; skipping locate/uninstall\n");
    }

    Print (L"-- InstallProtocolInterface test end --\n\n");
  }

  //
  // Ensure ACPI protocol available (real or stub) and install minimal RSDP into System Table
  //
  {
    EFI_STATUS AcpiStatus;
    EFI_ACPI_TABLE_PROTOCOL *AcpiProtocol = NULL;
    VOID *RsdpPtr = NULL;

    AcpiStatus = EnsureAcpiTableProtocolAvailable (&AcpiProtocol);
    Print (L"EnsureAcpiTableProtocolAvailable -> %r  Protocol=%p\n", AcpiStatus, AcpiProtocol);

    // Try to prepare+install an RSDP so shell tools (dmem) will show ACPI addresses
    AcpiStatus = PrepareAndInstallMinimalRsdp (&RsdpPtr, 36, SystemTable);
    Print (L"PrepareAndInstallMinimalRsdp -> %r  Rsdp=%p\n", AcpiStatus, RsdpPtr);


	//
	// Safe: print SystemTable config entries and validate pointers before use.
	// Uses both Print() (console) and DEBUG() (DebugLib event log).
	//
	{
	  BOOLEAN Found = FALSE;
	  UINTN ConfigCount = SystemTable->NumberOfTableEntries;

	  // Print summary
	  Print (L"\nSystemTable->NumberOfTableEntries = %u\n", (UINT32)ConfigCount);
	  DEBUG ((DEBUG_INFO, "SysTab entries=%u, SystemTable=%p\n", (UINT32)ConfigCount, SystemTable));

	  for (Index = 0; Index < ConfigCount; ++Index) {
		EFI_GUID *EntryGuid = &SystemTable->ConfigurationTable[Index].VendorGuid;
		VOID     *VendorTable = SystemTable->ConfigurationTable[Index].VendorTable;

		// Print the GUID entry index + pointer (safe)
		Print (L"ConfigTable[%u] GUID=%g  VendorTable=%p\n",
			   (UINT32)Index,
			   EntryGuid,
			   VendorTable);
		DEBUG ((DEBUG_INFO,
				"ConfigTable[%u] VendorTable ptr=%p, Guid pointer=%p\n",
				(UINT32)Index,
				VendorTable,
				EntryGuid));

		// Check if this entry is ACPI 2.0 or ACPI (legacy)
		if (CompareMem (EntryGuid, &gEfiAcpi20TableGuid, sizeof (EFI_GUID)) == 0) {
		  Print (L"  -> Found gEfiAcpi20TableGuid at index %u (VendorTable=%p)\n", (UINT32)Index, VendorTable);
		  DEBUG ((DEBUG_INFO, "  -> Found ACPI20 at index %u VendorTable=%p\n", (UINT32)Index, VendorTable));
		  Found = TRUE;

		  // Defensive: ensure VendorTable is not NULL before any deref
		  if (VendorTable == NULL) {
			Print (L"  WARNING: VendorTable is NULL for gEfiAcpi20TableGuid (skipping read)\n");
			DEBUG ((DEBUG_ERROR, "VendorTable NULL for ACPI20 at index %u\n", (UINT32)Index));
		  } else {
			// VendorTable came from SystemTable; but be defensive and only read minimal bytes
			// that are safe because we (should) have allocated the RSDP ourselves earlier.
			// If VendorTable == RsdpPtr (our buffer) it's safe to read; else we only print pointer.
			if (VendorTable == RsdpPtr) {
			  // Safe to examine our own RSDP content
			  CHAR8 sig8[9];
			  SetMem (sig8, sizeof(sig8), 0);
			  CopyMem (sig8, (CONST VOID *) VendorTable, 8);
			  // Convert to CHAR16 for safe Print()
			  CHAR16 sig16[9];
			  for (UINTN k = 0; k < 8; ++k) {
				sig16[k] = (CHAR16) sig8[k];
			  }
			  sig16[8] = 0;
			  Print (L"  RSDP (our buffer) signature='%s' at %p\n", sig16, VendorTable);
			  DEBUG ((DEBUG_INFO, "  RSDP signature=%a at %p\n", sig8, VendorTable));
			} else {
			  // VendorTable not our buffer — avoid dereferencing arbitrary pointers.
			  Print (L"  VendorTable not allocated by this app (skipping direct read)\n");
			  DEBUG ((DEBUG_WARN, "VendorTable %p not owned by app, skipping deref\n", VendorTable));
			}
		  }
		} else if (CompareMem (EntryGuid, &gEfiAcpiTableGuid, sizeof (EFI_GUID)) == 0) {
		  Print (L"  -> Found gEfiAcpiTableGuid (legacy RSDP v1) at index %u (VendorTable=%p)\n", (UINT32)Index, VendorTable);
		  DEBUG ((DEBUG_INFO, "  -> Found ACPI (v1) at index %u VendorTable=%p\n", (UINT32)Index, VendorTable));
		  Found = TRUE;

		  if (VendorTable == NULL) {
			Print (L"  WARNING: VendorTable is NULL for gEfiAcpiTableGuid (skipping read)\n");
			DEBUG ((DEBUG_ERROR, "VendorTable NULL for ACPI(v1) at index %u\n", (UINT32)Index));
		  } else {
			if (VendorTable == RsdpPtr) {
			  CHAR8 sig8[9];
			  SetMem (sig8, sizeof(sig8), 0);
			  CopyMem (sig8, (CONST VOID *) VendorTable, 8);
			  CHAR16 sig16[9];
			  for (UINTN k = 0; k < 8; ++k) sig16[k] = (CHAR16) sig8[k];
			  sig16[8] = 0;
			  Print (L"  RSDP (our buffer) signature='%s' at %p\n", sig16, VendorTable);
			  DEBUG ((DEBUG_INFO, "  RSDP signature=%a at %p\n", sig8, VendorTable));
			} else {
			  Print (L"  VendorTable not allocated by this app (skipping direct read)\n");
			  DEBUG ((DEBUG_WARN, "VendorTable %p not owned by app, skipping deref\n", VendorTable));
			}
		  }
		}
	  } // for Index

	  if (!Found) {
		Print (L"No ACPI GUID found in System Configuration Table.\n");
		DEBUG ((DEBUG_WARN, "No ACPI GUID found in SystemTable->ConfigurationTable\n"));
	  }

	  //
	  // Safe InstallAcpiTable test
	  // Only proceed if we have a valid AcpiProtocol pointer and our Rsdp buffer is present.
	  //
	  if (AcpiProtocol == NULL) {
		Print (L"Skipping InstallAcpiTable test: AcpiProtocol == NULL\n");
		DEBUG ((DEBUG_WARN, "AcpiProtocol is NULL, skipping InstallAcpiTable test\n"));
	  } else if (RsdpPtr == NULL) {
		Print (L"Skipping InstallAcpiTable test: RsdpPtr == NULL\n");
		DEBUG ((DEBUG_WARN, "RsdpPtr is NULL, nothing to install\n"));
	  } else {
		// We allocated RsdpPtr earlier via PrepareAndInstallMinimalRsdp, safe to use
		UINTN TableKey = 0;
		EFI_STATUS LocalStatus;

		Print (L"Attempting AcpiProtocol->InstallAcpiTable with RsdpPtr=%p ...\n", RsdpPtr);
		DEBUG ((DEBUG_INFO, "Calling InstallAcpiTable(AcpiProtocol=%p, RsdpPtr=%p)\n", AcpiProtocol, RsdpPtr));

		LocalStatus = AcpiProtocol->InstallAcpiTable (AcpiProtocol, RsdpPtr, 36, &TableKey);
		Print (L"AcpiProtocol->InstallAcpiTable -> %r (TableKey=%u)\n", LocalStatus, (UINT32)TableKey);
		DEBUG ((DEBUG_INFO, "InstallAcpiTable returned %r TableKey=%u\n", LocalStatus, (UINT32)TableKey));

		if (!EFI_ERROR (LocalStatus)) {
		  EFI_STATUS UnStatus = AcpiProtocol->UninstallAcpiTable (AcpiProtocol, TableKey);
		  Print (L"AcpiProtocol->UninstallAcpiTable(TableKey=%u) -> %r\n", (UINT32)TableKey, UnStatus);
		  DEBUG ((DEBUG_INFO, "UninstallAcpiTable -> %r\n", UnStatus));
		  if (EFI_ERROR (UnStatus)) {
			Print (L"Warning: UninstallAcpiTable failed -> %r\n", UnStatus);
			DEBUG ((DEBUG_ERROR, "UninstallAcpiTable failed -> %r\n", UnStatus));
		  }
		} else {
		  Print (L"InstallAcpiTable failed; implementation may validate or reject minimal headers.\n");
		  DEBUG ((DEBUG_WARN, "InstallAcpiTable failed -> %r\n", LocalStatus));
		}
	  } // end safe Install test
	}


	/*
    //
    // Now check System Table entries for ACPI GUIDs and print them
    //
    {
      BOOLEAN Found = FALSE;
      for (Index = 0; Index < SystemTable->NumberOfTableEntries; ++Index) {
        if (CompareMem (&SystemTable->ConfigurationTable[Index].VendorGuid, &gEfiAcpi20TableGuid, sizeof(EFI_GUID)) == 0) {
          Print (L"ConfigTable[%u] gEfiAcpi20TableGuid -> %p\n", (UINT32)Index, SystemTable->ConfigurationTable[Index].VendorTable);
          Found = TRUE;
        } else if (CompareMem (&SystemTable->ConfigurationTable[Index].VendorGuid, &gEfiAcpiTableGuid, sizeof(EFI_GUID)) == 0) {
          Print (L"ConfigTable[%u] gEfiAcpiTableGuid -> %p\n", (UINT32)Index, SystemTable->ConfigurationTable[Index].VendorTable);
          Found = TRUE;
        }
      }
      if (!Found) {
        Print (L"No ACPI GUID found in System Configuration Table.\n");
      }
    }

    //
    // Attempt a minimal ACPI header Install/Uninstall via AcpiProtocol if present
    //
    if (AcpiProtocol != NULL) {
      EFI_ACPI_DESCRIPTION_HEADER *Hdr = NULL;
      UINTN HdrSize = sizeof (EFI_ACPI_DESCRIPTION_HEADER);
      EFI_STATUS TestStatus;

      TestStatus = gBS->AllocatePool (EfiBootServicesData, HdrSize, (VOID **)&Hdr);
      if (EFI_ERROR (TestStatus) || Hdr == NULL) {
        Print (L"AllocatePool for test header failed -> %r\n", TestStatus);
      } else {
        SetMem ((VOID *) Hdr, HdrSize, 0);

        CopyMem ((VOID *) &Hdr->Signature, (CONST VOID *) "TST1", 4);
        Hdr->Length = (UINT32) HdrSize;
        Hdr->Revision = 1;
        CopyMem ((VOID *) Hdr->OemId, (CONST VOID *) "EDKEMU", 6);
        CopyMem ((VOID *) Hdr->OemTableId, (CONST VOID *) "EDK2TST ", 8);
        Hdr->OemRevision = 1;
        CopyMem ((VOID *) &Hdr->CreatorId, (CONST VOID *) "EDK2", 4);
        Hdr->CreatorRevision = 0x00000001;

        Hdr->Checksum = ComputeChecksum8 (Hdr, Hdr->Length);

        Print (L"Prepared test ACPI header at %p (size=%u)\n", Hdr, (UINT32)HdrSize);

        {
          UINTN TableKey = 0;
          TestStatus = AcpiProtocol->InstallAcpiTable (AcpiProtocol, Hdr, HdrSize, &TableKey);
          Print (L"AcpiProtocol->InstallAcpiTable -> %r (TableKey=%u)\n", TestStatus, (UINT32)TableKey);

          if (!EFI_ERROR (TestStatus)) {
            EFI_STATUS UnStatus = AcpiProtocol->UninstallAcpiTable (AcpiProtocol, TableKey);
            Print (L"AcpiProtocol->UninstallAcpiTable(TableKey=%u) -> %r\n", (UINT32)TableKey, UnStatus);
            if (EFI_ERROR (UnStatus)) {
              Print (L"Warning: UninstallAcpiTable failed -> %r\n", UnStatus);
            }
          } else {
            Print (L"InstallAcpiTable failed; implementation may validate or reject minimal headers.\n");
          }
        }

        gBS->FreePool (Hdr);
      }
    } else {
      Print (L"ACPI protocol not available (even stub); skipping InstallAcpiTable test\n");
    }
	*/
	
	
	
	
	
	
  } // end ACPI block

  Print (L"=== HelloBootServicesTest: end ===\n\n");
  return EFI_SUCCESS;
}
