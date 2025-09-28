/** UiApp.c
  Simple UEFI GUI app with keyboard menu and corner animation.
  - Uses Graphics Output Protocol (GOP) Blt operations for drawing
  - Calls an external test function UefiMain3(...) (your provided HelloBootServicesTest)
**/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiApplicationEntryPoint.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/PrintLib.h>
#include <Protocol/GraphicsOutput.h>
#include <Protocol/SimpleTextIn.h>

#define SPRITE_W 16
#define SPRITE_H 16
#define ANIM_FRAMES 4
#define ANIM_MS 200

// Prototype of the test you supplied earlier. Link the file with this module.
EFI_STATUS EFIAPI UefiMain3 (IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable);

STATIC EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop = NULL;

// Simple sprite frames buffer (each is SPRITE_W x SPRITE_H EFI_GRAPHICS_OUTPUT_BLT_PIXEL)
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL *gSpriteFrames[ANIM_FRAMES];

// Helper: allocate and build a set of simple frames (colored square with different patterns)
STATIC
EFI_STATUS
CreateSpriteFrames(VOID)
{
  UINTN i, x, y;
  for (i = 0; i < ANIM_FRAMES; ++i) {
    UINTN count = SPRITE_W * SPRITE_H;
    gSpriteFrames[i] = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL *)AllocateZeroPool(sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL) * count);
    if (gSpriteFrames[i] == NULL) {
      return EFI_OUT_OF_RESOURCES;
    }

    // Fill with a pattern that changes by frame index
    for (y = 0; y < SPRITE_H; ++y) {
      for (x = 0; x < SPRITE_W; ++x) {
        UINTN idx = y * SPRITE_W + x;
        EFI_GRAPHICS_OUTPUT_BLT_PIXEL *p = &gSpriteFrames[i][idx];

        // Simple varying color: base + frame offset
        UINT8 r = (UINT8)((x * 16 + i * 30) & 0xFF);
        UINT8 g = (UINT8)((y * 16 + i * 60) & 0xFF);
        UINT8 b = (UINT8)((x * 8 + y * 8 + i * 90) & 0xFF);

        // Create a simple alpha-like effect using checker
        if (((x + y + i) & 3) == 0) {
          p->Red   = r;
          p->Green = g;
          p->Blue  = b;
        } else {
          // partially transparent-like: mix with dark background
          p->Red   = r / 2;
          p->Green = g / 2;
          p->Blue  = b / 2;
        }
      }
    }
  }
  return EFI_SUCCESS;
}

STATIC
VOID
FreeSpriteFrames(VOID)
{
  UINTN i;
  for (i = 0; i < ANIM_FRAMES; ++i) {
    if (gSpriteFrames[i]) {
      FreePool(gSpriteFrames[i]);
      gSpriteFrames[i] = NULL;
    }
  }
}

// Draw textual menu using console. selectedIndex highlighted with '>'
STATIC
VOID
DrawMenu(CHAR16 **Items, UINTN ItemCount, UINTN Selected)
{
  UINTN i;
  // Clear text screen for clarity (console text)
  gST->ConOut->ClearScreen(gST->ConOut);
  for (i = 0; i < ItemCount; ++i) {
    if (i == Selected) {
      Print(L"> %s\n", Items[i]);
    } else {
      Print(L"  %s\n", Items[i]);
    }
  }
  Print(L"\nUse Up/Down to move, Enter to select, Esc to exit item or quit.\n");
}

// Animation mode: show sprite in corner, animate until Esc or Enter pressed
STATIC
VOID
AnimationMode(EFI_HANDLE ImageHandle)
{
  EFI_STATUS Status;
  EFI_EVENT TimerEvent = NULL;
  EFI_EVENT KeyEvent = gST->ConIn->WaitForKey;
  EFI_EVENT WaitList[2];
  UINTN WaitIndex;
  UINTN frame = 0;
  UINTN destX = 10, destY = 10;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL *bgBuffer = NULL;

  UINTN Width = SPRITE_W;
  UINTN Height = SPRITE_H;

  // Save background
  bgBuffer = AllocateZeroPool(sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL) * Width * Height);
  if (bgBuffer == NULL) {
    Print(L"Animation: cannot allocate bg buffer\n");
    return;
  }

  // Create periodic timer
  Status = gBS->CreateEvent(EVT_TIMER, TPL_APPLICATION, NULL, NULL, &TimerEvent);
  if (EFI_ERROR(Status)) {
    Print(L"Animation: CreateEvent Timer failed: %r\n", Status);
    FreePool(bgBuffer);
    return;
  }
  Status = gBS->SetTimer(TimerEvent, TimerPeriodic, ANIM_MS * 10000ULL /*100ns units*/);
  if (EFI_ERROR(Status)) {
    Print(L"Animation: SetTimer failed: %r\n", Status);
    gBS->CloseEvent(TimerEvent);
    FreePool(bgBuffer);
    return;
  }

  // First, capture background once
  Status = Gop->Blt(
            Gop,
            bgBuffer,
            EfiBltVideoToBltBuffer,
            destX, destY,
            0, 0,
            Width, Height,
            0);
  if (EFI_ERROR(Status)) {
    Print(L"Animation: initial Blt (save bg) failed: %r\n", Status);
    gBS->SetTimer(TimerEvent, TimerPeriodic, 0);
    gBS->CloseEvent(TimerEvent);
    FreePool(bgBuffer);
    return;
  }

  // Wait loop: TimerEvent + KeyEvent
  WaitList[0] = TimerEvent;
  WaitList[1] = KeyEvent;

  while (TRUE) {
    Status = gBS->WaitForEvent(2, WaitList, &WaitIndex);
    if (EFI_ERROR(Status)) break;

    if (WaitIndex == 0) {
      // Timer tick: advance frame
      // Restore background (so we don't overdraw old sprite)
      Status = Gop->Blt(Gop, bgBuffer, EfiBltBufferToVideo, 0,0, destX,destY, Width,Height, 0);
      if (EFI_ERROR(Status)) break;

      // Draw new frame
      Status = Gop->Blt(Gop, gSpriteFrames[frame], EfiBltBufferToVideo, 0,0, destX,destY, Width,Height, 0);
      if (EFI_ERROR(Status)) break;

      frame = (frame + 1) % ANIM_FRAMES;
    } else if (WaitIndex == 1) {
      // Key pressed: read it
      EFI_INPUT_KEY Key;
      Status = gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
      if (EFI_ERROR(Status)) {
        // ignore
      } else {
        if (Key.ScanCode == SCAN_ESC) {
          // Exit animation - restore bg and return
          Gop->Blt(Gop, bgBuffer, EfiBltBufferToVideo, 0,0, destX,destY, Width,Height, 0);
          break;
        }
        if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN || Key.UnicodeChar == CHAR_LINEFEED) {
          // Enter: also exit
          Gop->Blt(Gop, bgBuffer, EfiBltBufferToVideo, 0,0, destX,destY, Width,Height, 0);
          break;
        }
        // otherwise ignore
      }
    }
  }

  // Cleanup timer
  gBS->SetTimer(TimerEvent, TimerPeriodic, 0);
  gBS->CloseEvent(TimerEvent);
  if (bgBuffer) FreePool(bgBuffer);
}

// App entry
EFI_STATUS
EFIAPI
UefiMain3 (
  IN EFI_HANDLE ImageHandle,
  IN EFI_SYSTEM_TABLE *SystemTable
  )
{
  EFI_STATUS Status;
  CHAR16 *MenuItems[] = {
    L"Run HelloBootServicesTest (HelloBootServicesTest / your provided test)",
    L"Show corner animation",
    L"Exit"
  };
  UINTN ItemCount = sizeof(MenuItems) / sizeof(MenuItems[0]);
  UINTN Selected = 0;
  EFI_INPUT_KEY Key;

  // locate GOP (optional)
  Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&Gop);
  if (EFI_ERROR(Status) || Gop == NULL) {
    Print(L"Warning: GOP not available. Running text-only. Status=%r\n", Status);
  } else {
    // Prepare sprite frames
    Status = CreateSpriteFrames();
    if (EFI_ERROR(Status)) {
      Print(L"Warning: cannot allocate sprite frames: %r\n", Status);
    }
  }

  while (TRUE) {
    DrawMenu(MenuItems, ItemCount, Selected);

    // Wait for key
    Status = gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
    if (EFI_ERROR(Status)) {
      // If no key yet, wait
      UINTN Index;
      gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &Index);
      gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
    }

    if (Key.ScanCode == SCAN_UP) {
      if (Selected > 0) Selected--;
    } else if (Key.ScanCode == SCAN_DOWN) {
      if (Selected + 1 < ItemCount) Selected++;
    } else if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN || Key.UnicodeChar == CHAR_LINEFEED) {
      // Enter pressed: act on selection
      if (Selected == 0) {
        // Call your test (UefiMain3) which prints to console and does boot services tests
        Print(L"\n--- Entering HelloBootServicesTest ---\n\n");
        // Call it directly; it expects (ImageHandle, SystemTable)
        UefiMain3(ImageHandle, SystemTable);
        Print(L"\n--- Returned from HelloBootServicesTest ---\n\n");
        Print(L"Press any key to continue...\n");
        UINTN idx;
        gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &idx);
        // consume key
        gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
      } else if (Selected == 1) {
        // Animation
        if (Gop && gSpriteFrames[0]) {
          AnimationMode(ImageHandle);
          Print(L"\nAnimation ended. Press any key to continue...\n");
          UINTN idx;
          gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &idx);
          gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
        } else {
          Print(L"Cannot run animation: GOP not available or frames missing.\n");
          Print(L"Press any key to continue...\n");
          UINTN idx;
          gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &idx);
          gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
        }
      } else if (Selected == 2) {
        // Exit
        break;
      }
    } else if (Key.ScanCode == SCAN_ESC) {
      // Exit overall
      break;
    }
  }

  FreeSpriteFrames();
  Print(L"UiApp exiting.\n");
  return EFI_SUCCESS;
}
