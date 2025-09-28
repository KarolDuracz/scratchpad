/** HDMI-detect.c
  Minimal UEFI application to enumerate GOP handles + EDID-discovered displays,
  print basic information and provide a safe CpuDeadLoop wrapper.

  Build under EDK2 as a regular Application.
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/PrintLib.h>
#include <Library/BaseLib.h>
#include <Library/DebugLib.h>
#include <Library/MemoryAllocationLib.h>

#include <Protocol/GraphicsOutput.h>
#include <Protocol/EdidDiscovered.h>

#define SAFE_DEADLOOP_TIMEOUT_MS 5000    // wait 5 seconds for keypress before final CpuDeadLoop

/**
  Wait for a key for up to TimeoutMs. Return TRUE if a key was pressed, FALSE if timeout.
*/
STATIC
BOOLEAN
WaitForKeyOrTimeoutMs (
  IN UINTN TimeoutMs
  )
{
  EFI_STATUS Status;
  EFI_INPUT_KEY Key;
  UINTN Elapsed = 0;
  // Poll every 50ms
  while (Elapsed < TimeoutMs) {
    Status = gST->ConIn->ReadKeyStroke (gST->ConIn, &Key);
    if (Status == EFI_SUCCESS) {
      return TRUE;
    }
    gBS->Stall (50 * 1000); // 50 ms
    Elapsed += 50;
  }
  return FALSE;
}

/**
  Safely enter a dead loop: print message, wait for brief timeout allowing user to interrupt.
  If not interrupted, disable interrupts and call CpuDeadLoop().
*/
STATIC
VOID
SafeCpuDeadLoop (
  IN CHAR16 *Message
  )
{
  if (Message != NULL) {
    Print (L"%s\n", Message);
  } else {
    Print (L"Entering dead loop (press any key to abort)...\n");
  }

  Print (L"Press any key within %d ms to continue.\n", SAFE_DEADLOOP_TIMEOUT_MS);

  if (WaitForKeyOrTimeoutMs (SAFE_DEADLOOP_TIMEOUT_MS)) {
    Print (L"Key pressed — continuing normally.\n");
    return;
  }

  // Final: disable interrupts and spin forever in a controlled way
  // AsmDisableInterrupts is provided by BaseLib on supported architectures.
  // #if defined(__GNUC__) || defined(_MSC_VER)
  // BaseLib prototype:
  //VOID AsmDisableInterrupts(VOID); // error C2220
  //#endif

  /* Disable interrupts if available — best-effort; prototypes come from BaseLib */
  //AsmDisableInterrupts (); // error C2220

  Print (L"No keypress detected; entering CpuDeadLoop().\n");

  CpuDeadLoop (); // never returns
}

/**
  Minimal EDID parse for a few useful fields (manufacturer, product id, serial,
  week/year and monitor name if present). Safe: checks size >= 128.
*/
STATIC
VOID
ParseAndPrintEdid (
  IN UINT8  *Edid,
  IN UINTN   Size,
  IN UINTN   Index
  )
{
  if (Edid == NULL || Size < 128) {
    Print (L"EDID[%u]: invalid or too small (size=%u)\n", Index, (UINT32)Size);
    return;
  }

  // Manufacturer ID (bytes 8-9): 16-bit packed 5/5/6 -> decode into 3 ASCII letters
  UINT16 MfgCode = (Edid[8] << 8) | Edid[9];
  CHAR8 mfg[4] = {0,0,0,0};
  mfg[0] = (CHAR8)(((MfgCode >> 10) & 0x1F) ? ((MfgCode >> 10) & 0x1F) - 1 + 'A' : '?');
  mfg[1] = (CHAR8)(((MfgCode >> 5)  & 0x1F) ? ((MfgCode >> 5)  & 0x1F) - 1 + 'A' : '?');
  mfg[2] = (CHAR8)(((MfgCode >> 0)  & 0x1F) ? ((MfgCode >> 0)  & 0x1F) - 1 + 'A' : '?');

  UINT16 ProductId = (Edid[11] << 8) | Edid[10]; // product id (little-endian in EDID spec)
  UINT32 Serial = (Edid[15] << 24) | (Edid[14] << 16) | (Edid[13] << 8) | Edid[12];
  UINT8 Week = Edid[16];
  UINT8 Year = Edid[17]; // year offset from 1990

  CHAR16 MonitorName[64] = L"(not found)";

  // Descriptor blocks: 4 blocks of 18 bytes starting at offset 54 (0x36)
  for (UINTN d = 0; d < 4; d++) {
    UINTN off = 54 + d * 18;
    if (off + 18 > 128) {
      break;
    }
    // Descriptor block tag at bytes 3..4: if bytes 3==0 && 4==0 then it's a descriptor.
    // The tag (type) is at byte 3 and the content starts at byte 5 (index off+5) for ASCII strings.
    if (Edid[off] == 0x00 && Edid[off+1] == 0x00 && Edid[off+2] == 0x00) {
      UINT8 tag = Edid[off + 3];
      if (tag == 0xFC) {
        // Monitor name (ASCII), bytes off+5..off+17 (13 bytes)
        CHAR8 nameAscii[14] = {0};
        for (UINTN i = 0; i < 13; i++) {
          CHAR8 c = (CHAR8)Edid[off + 5 + i];
          nameAscii[i] = (c == 0x0A || c == 0x00) ? 0 : c;
        }
        // Convert ASCII to CHAR16
        for (UINTN i = 0; i < 13; i++) {
          if (nameAscii[i] == 0) break;
          MonitorName[i] = (CHAR16)nameAscii[i];
        }
        MonitorName[13] = L'\0';
        break;
      }
    }
  }

  Print (L"EDID[%u] Manufacturer=%a ProductId=0x%04x Serial=0x%08x Week=%u Year=%u Name=%s\n",
         Index, mfg, ProductId, Serial, Week, (UINT32)1990 + Year, MonitorName);
}

/**
  Application entry point.
*/
EFI_STATUS
EFIAPI
UefiMain3 (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;

  Print (L"EDK2 HDMI/Display probe demo\n");
  Print (L"---------------------------------------\n");

  //
  // 1) Enumerate GOP handles
  //
  UINTN GopHandleCount = 0;
  EFI_HANDLE *GopHandleBuffer = NULL;

  Status = gBS->LocateHandleBuffer (
                  ByProtocol,
                  &gEfiGraphicsOutputProtocolGuid,
                  NULL,
                  &GopHandleCount,
                  &GopHandleBuffer
                  );

  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(GOP) failed: %r\n", Status);
    // Allow user to inspect then abort into safe dead loop
    SafeCpuDeadLoop (L"Cannot find Graphics Output Protocol handles — aborting.");
    return Status;
  }

  Print (L"GOP handles found: %u\n", (UINT32)GopHandleCount);

  for (UINTN i = 0; i < GopHandleCount; i++) {
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop = NULL;
    Status = gBS->HandleProtocol (GopHandleBuffer[i], &gEfiGraphicsOutputProtocolGuid, (VOID **)&Gop);
    if (EFI_ERROR (Status) || Gop == NULL) {
      Print (L"  GOP[%u]: HandleProtocol failed: %r\n", (UINT32)i, Status);
      continue;
    }

    Print (L"\nGOP[%u] @ %p\n", (UINT32)i, GopHandleBuffer[i]);
    Print (L"  CurrentMode    : %u\n", (UINT32)Gop->Mode->Mode);
    Print (L"  MaxMode        : %u\n", (UINT32)Gop->Mode->MaxMode);
    Print (L"  Resolution     : %ux%u\n",
           (UINT32)Gop->Mode->Info->HorizontalResolution,
           (UINT32)Gop->Mode->Info->VerticalResolution);
    Print (L"  PixelFormat    : %u (0=PixelRedGreenBlueReserved8BitPerColor,...)\n", (UINT32)Gop->Mode->Info->PixelFormat);
    Print (L"  PixelsPerScan  : %u\n", (UINT32)Gop->Mode->Info->PixelsPerScanLine);
    Print (L"  FrameBufferBase: 0x%016lx\n", (UINT64)Gop->Mode->FrameBufferBase);
    Print (L"  FrameBufferSize: 0x%016lx bytes\n", (UINT64)Gop->Mode->FrameBufferSize);

    // enumerate modes
    for (UINT32 m = 0; m < Gop->Mode->MaxMode; m++) {
      EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *Info = NULL;
      UINTN InfoSize = 0;
      Status = Gop->QueryMode (Gop, m, &InfoSize, &Info);
      if (!EFI_ERROR (Status) && Info != NULL) {
        Print (L"    Mode %u: %ux%u (PixelsPerScanLine=%u) PixelFormat=%u\n",
               m,
               Info->HorizontalResolution,
               Info->VerticalResolution,
               Info->PixelsPerScanLine,
               Info->PixelFormat);
      }
    }
  }

  if (GopHandleBuffer != NULL) {
    FreePool (GopHandleBuffer);
  }

  //
  // 2) Enumerate EDID discovered protocol instances
  //
  UINTN EdidHandleCount = 0;
  EFI_HANDLE *EdidHandleBuffer = NULL;

  Status = gBS->LocateHandleBuffer (
                  ByProtocol,
                  &gEfiEdidDiscoveredProtocolGuid,
                  NULL,
                  &EdidHandleCount,
                  &EdidHandleBuffer
                  );

  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(EDID_DISCOVERED) returned %r (no EDID handles?)\n", Status);
    EdidHandleCount = 0;
  } else {
    Print (L"\nEDID discovered handles: %u\n", (UINT32)EdidHandleCount);
    for (UINTN i = 0; i < EdidHandleCount; i++) {
      EFI_EDID_DISCOVERED_PROTOCOL *EdidProto = NULL;
      Status = gBS->HandleProtocol (EdidHandleBuffer[i], &gEfiEdidDiscoveredProtocolGuid, (VOID **)&EdidProto);
      if (EFI_ERROR (Status) || EdidProto == NULL) {
        Print (L"  EDID[%u]: HandleProtocol failed: %r\n", (UINT32)i, Status);
        continue;
      }
      // EdidProto->Edid is pointer to EDID bytes, SizeOfEdid is total bytes (128 or 256..)
      UINT8 *EdidBytes = (UINT8 *)EdidProto->Edid;
      UINTN   SizeOfEdid = EdidProto->SizeOfEdid;
      Print (L"  EDID Handle %u @ %p size=%u bytes\n", (UINT32)i, EdidHandleBuffer[i], (UINT32)SizeOfEdid);
      ParseAndPrintEdid (EdidBytes, SizeOfEdid, i);
    }
  }

  if (EdidHandleBuffer != NULL) {
    FreePool (EdidHandleBuffer);
  }

  //
  // Final: summary and wait
  //
  Print (L"\nSummary: GOP handles = %u, EDID discovered entries = %u\n",
         (UINT32)GopHandleCount,
         (UINT32)EdidHandleCount);

  Print (L"\nPress any key to exit, or wait %d ms to enter final dead loop.\n", SAFE_DEADLOOP_TIMEOUT_MS);

  // Wait for key or enter safe dead loop
  if (!WaitForKeyOrTimeoutMs (SAFE_DEADLOOP_TIMEOUT_MS)) {
    SafeCpuDeadLoop (L"No key pressed — final action.");
    // SafeCpuDeadLoop may never return
  }

  // Clean exit
  Print (L"Exiting application.\n");

  return EFI_SUCCESS;
}
