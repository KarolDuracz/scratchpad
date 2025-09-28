/** HDMI-detect-display-test.c
  EDK2 UEFI application:
  - Enumerates GOP & EDID
  - Draws white background
  - Runs 3 color tests + animation
  - Runs 10s color-loop, then waits 5s for key to avoid CpuDeadLoop
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

#define SAFE_DEADLOOP_TIMEOUT_MS 5000    // 5 seconds before entering CpuDeadLoop
#define TEST_SHORT_MS            2000    // each short color test duration (2s)
#define ANIM_DURATION_MS         2000    // animation duration
#define COLOR_LOOP_MS           10000    // 10s color loop
#define POLL_MS                   50    // key poll granularity

// Helpers: in UEFI pixel order is Blue, Green, Red, Reserved
STATIC CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL PixelWhite = { 0xFF, 0xFF, 0xFF, 0x00 };
STATIC CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL PixelBlack = { 0x00, 0x00, 0x00, 0x00 };
STATIC CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL PixelRed   = { 0x00, 0x00, 0xFF, 0x00 };
STATIC CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL PixelGreen = { 0x00, 0xFF, 0x00, 0x00 };
STATIC CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL PixelBlue  = { 0xFF, 0x00, 0x00, 0x00 };
STATIC CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL PixelYellow= { 0x00, 0xFF, 0xFF, 0x00 }; // green+red

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
  while (Elapsed < TimeoutMs) {
    Status = gST->ConIn->ReadKeyStroke (gST->ConIn, &Key);
    if (Status == EFI_SUCCESS) {
      return TRUE;
    }
    gBS->Stall (POLL_MS * 1000);
    Elapsed += POLL_MS;
  }
  return FALSE;
}

/**
  Final dead loop helper (no AsmDisableInterrupts call per request).
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

  Print (L"No keypress detected; entering CpuDeadLoop().\n");
  CpuDeadLoop ();
}

/* Minimal EDID parser (unchanged, extracts manufacturer/product/name) */
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

  UINT16 MfgCode = (Edid[8] << 8) | Edid[9];
  CHAR8 mfg[4] = {0,0,0,0};
  mfg[0] = (CHAR8)(((MfgCode >> 10) & 0x1F) ? ((MfgCode >> 10) & 0x1F) - 1 + 'A' : '?');
  mfg[1] = (CHAR8)(((MfgCode >> 5)  & 0x1F) ? ((MfgCode >> 5)  & 0x1F) - 1 + 'A' : '?');
  mfg[2] = (CHAR8)(((MfgCode >> 0)  & 0x1F) ? ((MfgCode >> 0)  & 0x1F) - 1 + 'A' : '?');

  UINT16 ProductId = (Edid[11] << 8) | Edid[10];
  UINT32 Serial = (Edid[15] << 24) | (Edid[14] << 16) | (Edid[13] << 8) | Edid[12];
  UINT8 Week = Edid[16];
  UINT8 Year = Edid[17];

  CHAR16 MonitorName[64] = L"(not found)";
  for (UINTN d = 0; d < 4; d++) {
    UINTN off = 54 + d * 18;
    if (off + 18 > 128) break;
    if (Edid[off] == 0x00 && Edid[off+1] == 0x00 && Edid[off+2] == 0x00) {
      UINT8 tag = Edid[off + 3];
      if (tag == 0xFC) {
        CHAR8 nameAscii[14] = {0};
        for (UINTN i = 0; i < 13; i++) {
          CHAR8 c = (CHAR8)Edid[off + 5 + i];
          nameAscii[i] = (c == 0x0A || c == 0x00) ? 0 : c;
        }
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

/* Fill entire screen with a color via GOP Blt */
STATIC
EFI_STATUS
FillScreen (
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop,
  IN CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL *Color
  )
{
  if (Gop == NULL || Color == NULL) {
    return EFI_INVALID_PARAMETER;
  }
  return Gop->Blt (Gop,
                   (EFI_GRAPHICS_OUTPUT_BLT_PIXEL *)Color,
                   EfiBltVideoFill,
                   0, 0,
                   0, 0,
                   Gop->Mode->Info->HorizontalResolution,
                   Gop->Mode->Info->VerticalResolution,
                   0);
}

/* Draw color bars vertically (left-to-right) */
STATIC
VOID
DrawColorBars (
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop
  )
{
  UINT32 w = Gop->Mode->Info->HorizontalResolution;
  UINT32 h = Gop->Mode->Info->VerticalResolution;
  UINT32 n = 6;
  UINT32 barW = (w + n - 1) / n;

  EFI_GRAPHICS_OUTPUT_BLT_PIXEL colors[6] = {
    PixelRed, PixelGreen, PixelBlue, PixelYellow, PixelWhite, PixelBlack
  };

  for (UINT32 i = 0; i < n; i++) {
    UINT32 x = i * barW;
    UINT32 drawW = ((i == n-1) ? (w - x) : barW);
    Gop->Blt (Gop, &colors[i], EfiBltVideoFill, 0, 0, x, 0, drawW, h, 0);
  }
}

/* Draw a grid of small RGB squares */
STATIC
VOID
DrawGrid (
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop
  )
{
  UINT32 w = Gop->Mode->Info->HorizontalResolution;
  UINT32 h = Gop->Mode->Info->VerticalResolution;
  const UINT32 cols = 8;
  const UINT32 rows = 6;
  UINT32 cellW = w / cols;
  UINT32 cellH = h / rows;

  EFI_GRAPHICS_OUTPUT_BLT_PIXEL palette[3] = { PixelRed, PixelGreen, PixelBlue };

  for (UINT32 r = 0; r < rows; r++) {
    for (UINT32 c = 0; c < cols; c++) {
      UINT32 x = c * cellW;
      UINT32 y = r * cellH;
      EFI_GRAPHICS_OUTPUT_BLT_PIXEL *color = &palette[(r + c) % 3];
      Gop->Blt (Gop, color, EfiBltVideoFill, 0, 0, x, y, cellW, cellH, 0);
    }
  }
}

/* Simple animation: moving colored box back and forth */
STATIC
VOID
AnimateBox (
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop,
  IN UINTN DurationMs
  )
{
  UINT32 w = Gop->Mode->Info->HorizontalResolution;
  UINT32 h = Gop->Mode->Info->VerticalResolution;

  const UINT32 boxW = (w / 10) ? (w / 10) : 50;
  const UINT32 boxH = (h / 10) ? (h / 10) : 30;
  UINTN elapsed = 0;
  INT32 direction = 1;
  UINT32 x = 0;

  // start with black background for animation clarity
  FillScreen (Gop, &PixelBlack);

  while (elapsed < DurationMs) {
    // clear previous by filling entire screen black then draw box (simple)
    FillScreen (Gop, &PixelBlack);

    // compute x position
    x += direction ? 8 : -8;
    if ((INT32)x + (INT32)boxW >= (INT32)w) {
      direction = 0;
      x = w - boxW;
    } else if ((INT32)x <= 0) {
      direction = 1;
      x = 0;
    }

    // rectangle color cycles with time
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL color = ( (elapsed / 100) % 3 == 0 ) ? PixelRed :
                                          ( (elapsed / 100) % 3 == 1 ) ? PixelGreen : PixelBlue;

    Gop->Blt (Gop, &color, EfiBltVideoFill, 0, 0, x, (h - boxH) / 2, boxW, boxH, 0);

    gBS->Stall (33 * 1000); // ~30 FPS
    elapsed += 33;
  }
}

/* Simple small helper to run three color tests and short animation on a given GOP */
STATIC
VOID
RunGraphicsTestsOnGop (
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop
  )
{
  if (Gop == NULL) return;

  // 1) White background
  FillScreen (Gop, &PixelWhite);
  gBS->Stall (300 * 1000); // short pause so the white is visible (0.3s)

  // 2) Test A: Solid fills (red, green, blue) each TEST_SHORT_MS
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL solidColors[3] = { PixelRed, PixelGreen, PixelBlue };
  for (UINTN i = 0; i < 3; i++) {
    FillScreen (Gop, &solidColors[i]);
    gBS->Stall (TEST_SHORT_MS * 1000);
  }

  // 3) Test B: Color bars
  DrawColorBars (Gop);
  gBS->Stall (TEST_SHORT_MS * 1000);

  // 4) Test C: Grid of RGB squares
  DrawGrid (Gop);
  gBS->Stall (TEST_SHORT_MS * 1000);

  // 5) Simple animation
  AnimateBox (Gop, ANIM_DURATION_MS);

  // 6) After animation, restore white background
  FillScreen (Gop, &PixelWhite);
}

/* Run a 10-second color loop (repeated color bars) */
STATIC
VOID
RunColorLoop10s (
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop
  )
{
  if (Gop == NULL) return;

  UINTN elapsed = 0;
  while (elapsed < COLOR_LOOP_MS) {
    DrawColorBars (Gop);
    gBS->Stall (500 * 1000); // show bars for 0.5s
    DrawGrid (Gop);
    gBS->Stall (500 * 1000);
    FillScreen (Gop, &PixelWhite);
    gBS->Stall (200 * 1000);
    elapsed += 1200; // approx
  }
}

/* Application entry point */
EFI_STATUS
EFIAPI
UefiMain3 (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;

  Print (L"EDK2 HDMI/Display probe + graphical tests\n");
  Print (L"------------------------------------------\n");

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

  if (EFI_ERROR (Status) || GopHandleCount == 0) {
    Print (L"LocateHandleBuffer(GOP) failed: %r (or zero handles)\n", Status);
    SafeCpuDeadLoop (L"Cannot find Graphics Output Protocol handles — aborting.");
    return Status;
  }

  Print (L"GOP handles found: %u\n", (UINT32)GopHandleCount);

  // We'll run graphical tests on the first GOP handle (console) and print info for all
  EFI_GRAPHICS_OUTPUT_PROTOCOL *PrimaryGop = NULL;
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
    Print (L"  PixelFormat    : %u\n", (UINT32)Gop->Mode->Info->PixelFormat);
    Print (L"  PixelsPerScan  : %u\n", (UINT32)Gop->Mode->Info->PixelsPerScanLine);
    Print (L"  FrameBufferBase: 0x%016lx\n", (UINT64)Gop->Mode->FrameBufferBase);
    Print (L"  FrameBufferSize: 0x%016lx bytes\n", (UINT64)Gop->Mode->FrameBufferSize);

    if (PrimaryGop == NULL) {
      PrimaryGop = Gop;
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
      UINT8 *EdidBytes = (UINT8 *)EdidProto->Edid;
      UINTN   SizeOfEdid = EdidProto->SizeOfEdid;
      Print (L"  EDID Handle %u @ %p size=%u bytes\n", (UINT32)i, EdidHandleBuffer[i], (UINT32)SizeOfEdid);
      ParseAndPrintEdid (EdidBytes, SizeOfEdid, i);
    }
  }

  if (EdidHandleBuffer != NULL) {
    FreePool (EdidHandleBuffer);
  }

  Print (L"\nSummary: GOP handles = %u, EDID discovered entries = %u\n",
         (UINT32)GopHandleCount,
         (UINT32)EdidHandleCount);

  //
  // If we have a primary GOP, run graphical tests on it
  //
  if (PrimaryGop != NULL) {
    Print (L"\nRunning graphical tests on primary GOP (press any key to skip tests)...\n");

    // allow user to interrupt starting the tests quickly
    if (!WaitForKeyOrTimeoutMs (300)) {
      // Run the short suite
      RunGraphicsTestsOnGop (PrimaryGop);

      // 10-second color test loop
      Print (L"\nRunning 10-second color-loop test...\n");
      RunColorLoop10s (PrimaryGop);

      // restore white background at end
      FillScreen (PrimaryGop, &PixelWhite);
    } else {
      Print (L"User pressed key — skipping graphical tests.\n");
    }
  } else {
    Print (L"No GOP available to run graphical tests.\n");
  }

  //
  // Final: Wait for user to press key (5s) to avoid deadloop, otherwise deadloop
  //
  Print (L"\nPress any key within %d ms to avoid final dead loop.\n", SAFE_DEADLOOP_TIMEOUT_MS);
  if (!WaitForKeyOrTimeoutMs (SAFE_DEADLOOP_TIMEOUT_MS)) {
    SafeCpuDeadLoop (L"No key pressed — final action.");
    // may not return
  }

  Print (L"Exiting application normally.\n");
  return EFI_SUCCESS;
}
