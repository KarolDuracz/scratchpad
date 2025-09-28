/** UiAppGui.c
  Refactored GUI-style UEFI app demo for EDK2.
  - GOP graphics
  - Simple UI core (screens stack)
  - Creature sprite (procedural)
  - Smoke particle demo
  - Calls external UefiMain3(...) (user test)
**/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiApplicationEntryPoint.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/PrintLib.h>
/* DON'T include TimerLib.h to avoid unresolved GetPerformanceCounter/GetTimeInNanoSecond */
#include <Protocol/GraphicsOutput.h>
#include <Protocol/SimpleTextIn.h>

//
// External test supplied by user - should be linked from HelloBootServicesTest.c
//
EFI_STATUS EFIAPI UefiMain3(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable);

// -------------------- Configuration --------------------
#define UI_MAX_STACK 8
#define UI_FPS_MS 80
#define CREATURE_W 24
#define CREATURE_H 24
#define SMOKE_MAX_PARTICLES 64

#ifdef _MSC_VER
// MSVC emits a reference to _fltused when floating-point is used.
// In freestanding builds (UEFI/EDK2) the CRT is not linked, so provide the symbol.
int _fltused = 0;
#endif

// -------------------- Typedefs / small helpers --------------------
typedef EFI_GRAPHICS_OUTPUT_BLT_PIXEL PIXEL;

STATIC EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop = NULL;
STATIC UINT32 ScreenWidth = 0;
STATIC UINT32 ScreenHeight = 0;

// tiny clamp helpers
STATIC
UINT8
ClampU8(UINT32 v)
{
  if (v > 255) return 255;
  return (UINT8)v;
}

STATIC
UINT32
ClampU32(UINT32 v, UINT32 min, UINT32 max)
{
  if (v < min) return min;
  if (v > max) return max;
  return v;
}

// Small helper to round a double to int (works for positive and negative)
STATIC
INT32
RoundDoubleToInt(double v)
{
  if (v >= 0.0) {
    return (INT32)(v + 0.5);
  } else {
    return (INT32)(v - 0.5);
  }
}

// -------------------- RNG (simple LCG) --------------------
typedef struct {
  UINT32 state;
} RNG_CTX;

STATIC
VOID
RngInitFromPerf(RNG_CTX *r)
{
  if (r == NULL) return;
  // Use UEFI timer via gBS->CreateEvent/SetTimer would be heavier; instead seed from current time
  // Use UEFI's GetTime is a runtime service and may not be present; use gBS->SetTimer trick is overkill.
  // As a simple seed, take the address of r and XOR it with UEFI SystemTable pointer (semi-random)
  UINTN seed = (UINTN)r ^ (UINTN)gST;
  r->state = (UINT32)(seed ^ 0xA5A5A5A5);
  if (r->state == 0) r->state = 0x1234567;
}

STATIC
UINT32
RngNext(RNG_CTX *r)
{
  // LCG constants from Numerical Recipes
  r->state = (UINT32)(1664525u * r->state + 1013904223u);
  return r->state;
}

// -------------------- GFX helpers --------------------
STATIC
EFI_STATUS
GfxInit(VOID)
{
  EFI_STATUS Status;
  Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&Gop);
  if (EFI_ERROR(Status) || Gop == NULL) {
    return EFI_NOT_FOUND;
  }
  ScreenWidth  = Gop->Mode->Info->HorizontalResolution;
  ScreenHeight = Gop->Mode->Info->VerticalResolution;
  return EFI_SUCCESS;
}

STATIC
VOID
GfxFillRect(UINT32 X, UINT32 Y, UINT32 W, UINT32 H, PIXEL Color)
{
  if (Gop == NULL) return;
  // allocate buffer of W*H pixels
  UINTN Count = (UINTN)W * (UINTN)H;
  PIXEL *Buf = AllocateZeroPool(sizeof(PIXEL) * Count);
  if (Buf == NULL) return;
  for (UINTN i = 0; i < Count; ++i) Buf[i] = Color;
  Gop->Blt(Gop, Buf, EfiBltBufferToVideo, 0, 0, X, Y, W, H, 0);
  FreePool(Buf);
}

STATIC
PIXEL
BlendPixel(PIXEL src, PIXEL dst, UINT8 alpha)
{
  PIXEL r;
  UINT32 inv = 255 - alpha;
  r.Red   = (UINT8)((src.Red   * alpha + dst.Red   * inv) / 255);
  r.Green = (UINT8)((src.Green * alpha + dst.Green * inv) / 255);
  r.Blue  = (UINT8)((src.Blue  * alpha + dst.Blue  * inv) / 255);
  return r;
}

/**
  GfxBlitBufferToVideo:
    - Buffer: pointer to PIXEL buffer of size BufW*BufH
    - Alpha: optional pointer to a byte-per-pixel alpha map (same size)
**/
STATIC
EFI_STATUS
GfxBlitBufferToVideo(PIXEL *Buffer, UINT32 BufW, UINT32 BufH, UINT32 X, UINT32 Y, UINT8 *Alpha)
{
  if (Gop == NULL || Buffer == NULL) return EFI_INVALID_PARAMETER;
  if (Alpha == NULL) {
    return Gop->Blt(Gop, Buffer, EfiBltBufferToVideo, 0, 0, X, Y, BufW, BufH, 0);
  }

  // need to read destination, blend and blit
  UINTN Count = (UINTN)BufW * (UINTN)BufH;
  PIXEL *Dst = AllocatePool(sizeof(PIXEL) * Count);
  PIXEL *Tmp = AllocatePool(sizeof(PIXEL) * Count);
  if (Dst == NULL || Tmp == NULL) { FreePool(Dst); FreePool(Tmp); return EFI_OUT_OF_RESOURCES; }

  EFI_STATUS Status = Gop->Blt(Gop, Dst, EfiBltVideoToBltBuffer, X, Y, 0, 0, BufW, BufH, 0);
  if (EFI_ERROR(Status)) { FreePool(Dst); FreePool(Tmp); return Status; }

  for (UINTN i = 0; i < Count; ++i) {
    UINT8 a = Alpha[i];
    if (a == 255) {
      Tmp[i] = Buffer[i];
    } else if (a == 0) {
      Tmp[i] = Dst[i];
    } else {
      Tmp[i] = BlendPixel(Buffer[i], Dst[i], a);
    }
  }

  Status = Gop->Blt(Gop, Tmp, EfiBltBufferToVideo, 0, 0, X, Y, BufW, BufH, 0);
  FreePool(Dst);
  FreePool(Tmp);
  return Status;
}

// -------------------- Sprite (creature) --------------------
typedef struct {
  UINT32 W;
  UINT32 H;
  UINTN FrameCount;
  PIXEL *Frames;   // frames contiguous: FrameCount * (W*H)
  UINT8  *Alpha;   // same layout, per-pixel alpha
} SPRITE;

STATIC SPRITE *
SpriteCreateProceduralCreature(UINTN FrameCount)
{
  UINT32 W = (UINT32)CREATURE_W;
  UINT32 H = (UINT32)CREATURE_H;
  if (FrameCount == 0) return NULL;

  SPRITE *s = AllocateZeroPool(sizeof(*s));
  if (!s) return NULL;

  UINTN pixPerFrame = (UINTN)W * (UINTN)H;
  s->Frames = AllocateZeroPool(sizeof(PIXEL) * pixPerFrame * FrameCount);
  s->Alpha  = AllocateZeroPool(sizeof(UINT8)  * pixPerFrame * FrameCount);
  if (!s->Frames || !s->Alpha) {
    FreePool(s->Frames);
    FreePool(s->Alpha);
    FreePool(s);
    return NULL;
  }

  s->W = W;
  s->H = H;
  s->FrameCount = FrameCount;

  // simple procedural artwork: circular body + eyes + animated tail
  for (UINTN f = 0; f < FrameCount; ++f) {
    for (UINT32 y = 0; y < H; ++y) {
      for (UINT32 x = 0; x < W; ++x) {
        UINTN idx = f * pixPerFrame + (UINTN)y * W + x;
        PIXEL *p = &s->Frames[idx];
        UINT8  *a = &s->Alpha[idx];
        p->Red = p->Green = p->Blue = 0;
        *a = 0;

        INT32 cx = (INT32)(W / 2);
        INT32 cy = (INT32)(H / 2);
        INT32 dx = (INT32)x - cx;
        INT32 dy = (INT32)y - cy;
        INT32 dist2 = dx*dx + dy*dy;
        INT32 rbody = (cx - 2);

        if (dist2 <= rbody * rbody) {
          // body base color varies by frame
          p->Red   = (UINT8)(160 + (f * 8));
          p->Green = (UINT8)(80  + (f * 6));
          p->Blue  = 64;
          *a = 255;
        }

        // eyes positions
        if ((INT32)y == cy - 3 && ((INT32)x == cx - 4 || (INT32)x == cx + 4)) {
          // blink on one frame: simple blink pattern
          if ((f % FrameCount) == 1) {
            p->Red = p->Green = p->Blue = 20;
            *a = 255;
          } else {
            p->Red = p->Green = p->Blue = 0;
            *a = 255;
          }
        }

        // tail animated: shifting vertical offset based on frame index
        {
          INT32 tailOffset = (INT32)f - (INT32)(FrameCount / 2);
          INT32 dyRelation = (INT32)y - (cy + tailOffset);
          INT32 absDyRelation = dyRelation < 0 ? -dyRelation : dyRelation;
          if ((INT32)x > cx + 3 && absDyRelation <= 1 && (INT32)x < (INT32)W - 2) {
            p->Red = 220; p->Green = 100; p->Blue = 50;
            *a = 220;
          }
        }
      }
    }
  }

  return s;
}

STATIC VOID SpriteFree(SPRITE *s)
{
  if (!s) return;
  if (s->Frames) FreePool(s->Frames);
  if (s->Alpha) FreePool(s->Alpha);
  FreePool(s);
}

// -------------------- Smoke particle system --------------------
typedef struct {
  double x, y;
  double vx, vy;
  double life;     // seconds remaining
  double maxlife;
} PARTICLE;

typedef struct {
  PARTICLE parts[SMOKE_MAX_PARTICLES];
  RNG_CTX rng;
} SMOKE_SYSTEM;

STATIC VOID SmokeInit(SMOKE_SYSTEM *ss)
{
  ZeroMem(ss, sizeof(*ss));
  RngInitFromPerf(&ss->rng);
}

STATIC VOID SmokeSpawn(SMOKE_SYSTEM *ss, double x, double y)
{
  if (!ss) return;
  for (UINTN i = 0; i < SMOKE_MAX_PARTICLES; ++i) {
    if (ss->parts[i].life <= 0.0) {
      // produce small random
      UINT32 r = RngNext(&ss->rng);
      double rr = (double)(r & 0xFFFF) / 65535.0;
      ss->parts[i].x = x + (rr - 0.5) * 12.0;
      ss->parts[i].y = y;
      ss->parts[i].vx = (rr - 0.5) * 0.5;
      ss->parts[i].vy = -0.3 - rr * 0.6;
      ss->parts[i].life = 0.6 + rr * 0.8;
      ss->parts[i].maxlife = ss->parts[i].life;
      break;
    }
  }
}

STATIC VOID SmokeUpdateAndRender(SMOKE_SYSTEM *ss, UINT32 px, UINT32 py)
{
  if (!ss || Gop == NULL) return;

  // simple local buffer size
  UINT32 W = 48, H = 48;
  INT32 ox = (INT32)px - (INT32)(W/2);
  INT32 oy = (INT32)py - (INT32)(H/2);

  UINTN Count = (UINTN)W * (UINTN)H;
  PIXEL *buf = AllocateZeroPool(sizeof(PIXEL) * Count);
  UINT8  *alpha = AllocateZeroPool(sizeof(UINT8) * Count);
  if (!buf || !alpha) { FreePool(buf); FreePool(alpha); return; }

  // update particles
  double dt = (double)UI_FPS_MS / 1000.0;
  for (UINTN i = 0; i < SMOKE_MAX_PARTICLES; ++i) {
    PARTICLE *p = &ss->parts[i];
    if (p->life > 0.0) {
      // integrate
      p->x += p->vx * dt * 60.0 * 0.016;
      p->y += p->vy * dt * 60.0 * 0.016;
      p->life -= dt;

      double norm = (p->maxlife > 0.0) ? (p->life / p->maxlife) : 0.0;
      if (norm < 0.0) norm = 0.0;
      UINT8 baseAlpha = (UINT8)ClampU8((UINT32)(255.0 * norm));

      INT32 cx = RoundDoubleToInt(p->x) - ox;
      INT32 cy = RoundDoubleToInt(p->y) - oy;

      for (INT32 yy = -6; yy <= 6; ++yy) {
        for (INT32 xx = -6; xx <= 6; ++xx) {
          INT32 rx = cx + xx;
          INT32 ry = cy + yy;
          if (rx < 0 || rx >= (INT32)W || ry < 0 || ry >= (INT32)H) continue;
          INT32 d2 = xx*xx + yy*yy;
          if (d2 > 36) continue;
          UINTN idx = (UINTN)ry * W + (UINTN)rx;

          PIXEL s;
          s.Red = s.Green = s.Blue = 180;

          UINT8 localAlpha = (UINT8)((baseAlpha * (36 - d2)) / 36);

          // blend into buffer by simple formula: result = blend(source, dst, localAlpha)
          PIXEL dst = buf[idx];
          buf[idx] = BlendPixel(s, dst, localAlpha);
          // set alpha map to brightness
          alpha[idx] = (UINT8)ClampU8((UINT32)((buf[idx].Red + buf[idx].Green + buf[idx].Blue) / 3));
        }
      }
    }
  }

  // blit composed buffer with alpha
  GfxBlitBufferToVideo(buf, W, H, (UINT32)ox, (UINT32)oy, alpha);

  FreePool(buf);
  FreePool(alpha);
}

// -------------------- UI core --------------------
typedef struct UI_SCREEN UI_SCREEN;
typedef EFI_STATUS (*UI_Action)(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable, IN VOID *Context);

typedef struct {
  CHAR16 *Title;
  CHAR16 *Description;
  UI_Action Action;
  VOID *Context;
  UI_SCREEN *SubScreen;
} UI_MenuItem;

struct UI_SCREEN {
  CHAR16 *Title;
  UI_MenuItem *Items;
  UINTN ItemCount;
  UINTN Selected;
};

STATIC UI_SCREEN *UiStack[UI_MAX_STACK];
STATIC INTN UiStackTop = -1;

STATIC VOID UiPush(UI_SCREEN *s)
{
  if (UiStackTop + 1 < UI_MAX_STACK) {
    UiStackTop++;
    UiStack[UiStackTop] = s;
  }
}

STATIC VOID UiPop(VOID)
{
  if (UiStackTop >= 0) {
    UiStack[UiStackTop] = NULL;
    UiStackTop--;
  }
}

STATIC UI_SCREEN *UiTop(VOID)
{
  if (UiStackTop >= 0) return UiStack[UiStackTop];
  return NULL;
}

// Forward declarations of action functions
STATIC EFI_STATUS Action_RunTest(IN EFI_HANDLE, IN EFI_SYSTEM_TABLE *, IN VOID *);
STATIC EFI_STATUS Action_CreatureDetail(IN EFI_HANDLE, IN EFI_SYSTEM_TABLE *, IN VOID *);
STATIC EFI_STATUS Action_SmokeDetail(IN EFI_HANDLE, IN EFI_SYSTEM_TABLE *, IN VOID *);

// UI draw: draw panels and textual overlays via console for simplicity
STATIC VOID UiDrawCurrent(IN UI_SCREEN *s)
{
  if (!s || Gop == NULL) return;

  // background
  PIXEL bg = { .Blue = 20, .Green = 30, .Red = 60 };
  GfxFillRect(0, 0, ScreenWidth, ScreenHeight, bg);

  // title bar
  PIXEL bar = { .Red = 10, .Green = 70, .Blue = 140 };
  GfxFillRect(0, 0, ScreenWidth, 40, bar);

  // draw title on console overlay
  gST->ConOut->SetCursorPosition(gST->ConOut, 1, 0);
  gST->ConOut->OutputString(gST->ConOut, s->Title);

  UINT32 leftW = ScreenWidth / 3;
  PIXEL panelBg = { .Red = 40, .Green = 40, .Blue = 60 };
  GfxFillRect(10, 50, leftW - 20, ScreenHeight - 60, panelBg);

  UINT32 rightX = leftW + 20;
  PIXEL infoBg = { .Red = 10, .Green = 10, .Blue = 18 };
  GfxFillRect(rightX, 50, ScreenWidth - rightX - 10, ScreenHeight - 60, infoBg);

  // draw menu items as text lines using console at approximate positions
  for (UINTN i = 0; i < s->ItemCount; ++i) {
    CHAR16 buf[256];
    UnicodeSPrint(buf, sizeof(buf), L" %c %s", (i == s->Selected) ? L'>' : L' ', s->Items[i].Title);
    gST->ConOut->SetCursorPosition(gST->ConOut, 2, (INT32)(3 + i));
    gST->ConOut->OutputString(gST->ConOut, buf);
  }

  // draw description / info for selected item on right panel
  CHAR16 *info = L"";
  if (s->Selected < s->ItemCount) {
    info = s->Items[s->Selected].Description ? s->Items[s->Selected].Description : L"";
  }
  UINTN infoCol = (rightX / 8) + 2;
  UINTN infoRow = 3;
  for (UINTN r = 0; r < 20; ++r) {
    gST->ConOut->SetCursorPosition(gST->ConOut, infoCol, infoRow + r);
    gST->ConOut->OutputString(gST->ConOut, L"                                                                                ");
  }
  gST->ConOut->SetCursorPosition(gST->ConOut, infoCol, infoRow);
  gST->ConOut->OutputString(gST->ConOut, info);

  // hint
  gST->ConOut->SetCursorPosition(gST->ConOut, 1, (UINTN)(ScreenHeight / 16) + 14);
  gST->ConOut->OutputString(gST->ConOut, L"Use Up/Down, Enter to open, Esc to go back / exit.");
}

// -------------------- Actions --------------------
STATIC EFI_STATUS Action_RunTest(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable, IN VOID *Context)
{
  // text mode for the test
  gST->ConOut->ClearScreen(gST->ConOut);
  Print(L"--- Running HelloBootServicesTest ---\n\n");
  UefiMain3(ImageHandle, SystemTable);
  Print(L"\n--- Test finished. Press any key to continue. ---\n");
  UINTN idx;
  gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &idx);
  EFI_INPUT_KEY k;
  gST->ConIn->ReadKeyStroke(gST->ConIn, &k);
  return EFI_SUCCESS;
}

STATIC EFI_STATUS Action_CreatureDetail(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable, IN VOID *Context)
{
  SPRITE *spr = (SPRITE *)Context;
  if (!spr) return EFI_INVALID_PARAMETER;

  EFI_EVENT TimerEvent = NULL;
  EFI_EVENT WaitList[2];
  UINTN WaitIndex;
  EFI_STATUS Status;
  UINTN frame = 0;

  Status = gBS->CreateEvent(EVT_TIMER, TPL_APPLICATION, NULL, NULL, &TimerEvent);
  if (EFI_ERROR(Status)) return Status;

  Status = gBS->SetTimer(TimerEvent, TimerPeriodic, (UINT64)UI_FPS_MS * 10000ULL);
  if (EFI_ERROR(Status)) { gBS->CloseEvent(TimerEvent); return Status; }

  WaitList[0] = TimerEvent;
  WaitList[1] = gST->ConIn->WaitForKey;

  while (TRUE) {
    // background
    PIXEL bg = { .Blue = 20, .Green = 30, .Red = 50 };
    GfxFillRect(0, 0, ScreenWidth, ScreenHeight, bg);

    // scale by 2
    UINT32 outW = spr->W * 2;
    UINT32 outH = spr->H * 2;
    PIXEL *buf = AllocateZeroPool(sizeof(PIXEL) * outW * outH);
    UINT8  *alph = AllocateZeroPool(sizeof(UINT8) * outW * outH);
    if (!buf || !alph) { FreePool(buf); FreePool(alph); break; }

    // nearest-neighbor scaling
    for (UINT32 y = 0; y < outH; ++y) {
      for (UINT32 x = 0; x < outW; ++x) {
        UINT32 sx = x / 2;
        UINT32 sy = y / 2;
        UINTN sidx = (frame % spr->FrameCount) * (spr->W * spr->H) + sy * spr->W + sx;
        UINTN idx = y * outW + x;
        buf[idx] = spr->Frames[sidx];
        alph[idx] = spr->Alpha[sidx];
      }
    }

    UINT32 px = (ScreenWidth - outW) / 2;
    UINT32 py = (ScreenHeight - outH) / 3;
    GfxBlitBufferToVideo(buf, outW, outH, px, py, alph);

    FreePool(buf);
    FreePool(alph);

    gST->ConOut->SetCursorPosition(gST->ConOut, 2, 2);
    gST->ConOut->OutputString(gST->ConOut, L"Creature Detail (Esc to go back)");

    Status = gBS->WaitForEvent(2, WaitList, &WaitIndex);
    if (EFI_ERROR(Status)) break;

    if (WaitIndex == 1) {
      EFI_INPUT_KEY Key;
      gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
      if (Key.ScanCode == SCAN_ESC) break;
      if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN) break;
    } else {
      frame = (frame + 1) % spr->FrameCount;
    }
  }

  gBS->SetTimer(TimerEvent, TimerPeriodic, 0);
  gBS->CloseEvent(TimerEvent);
  return EFI_SUCCESS;
}

STATIC EFI_STATUS Action_SmokeDetail(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable, IN VOID *Context)
{
  SMOKE_SYSTEM ss;
  SmokeInit(&ss);

  EFI_EVENT TimerEvent = NULL;
  EFI_EVENT WaitList[2];
  UINTN WaitIndex;
  EFI_STATUS Status = EFI_SUCCESS;

  Status = gBS->CreateEvent(EVT_TIMER, TPL_APPLICATION, NULL, NULL, &TimerEvent);
  if (EFI_ERROR(Status)) return Status;
  Status = gBS->SetTimer(TimerEvent, TimerPeriodic, (UINT64)UI_FPS_MS * 10000ULL);
  if (EFI_ERROR(Status)) { gBS->CloseEvent(TimerEvent); return Status; }

  WaitList[0] = TimerEvent;
  WaitList[1] = gST->ConIn->WaitForKey;

  while (TRUE) {
    Status = gBS->WaitForEvent(2, WaitList, &WaitIndex);
    if (EFI_ERROR(Status)) break;

    if (WaitIndex == 1) {
      EFI_INPUT_KEY Key;
      gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
      if (Key.ScanCode == SCAN_ESC) break;
    } else {
      // spawn and render smoke at bottom center
      SmokeSpawn(&ss, (double)(ScreenWidth / 2), (double)(ScreenHeight * 3 / 4));
      SmokeUpdateAndRender(&ss, ScreenWidth / 2, (ScreenHeight * 3 / 4));
    }
  }

  gBS->SetTimer(TimerEvent, TimerPeriodic, 0);
  gBS->CloseEvent(TimerEvent);
  return EFI_SUCCESS;
}

// -------------------- Main app --------------------
STATIC SPRITE *gCreature = NULL;

STATIC UI_MenuItem MainMenuItems[] = {
  { L"Run HelloBootServicesTest", L"Run the Boot Services test (timers, protocols, memory pool).", Action_RunTest, NULL, NULL },
  { L"Creature (preview)",        L"Small creature animation. Enter to open detail view.", NULL, NULL, NULL },
  { L"Smoke effect demo",         L"Simulated smoke particle system. Enter to open.", NULL, NULL, NULL },
  { L"Exit",                      L"Exit this application", NULL, NULL, NULL }
};

STATIC UI_SCREEN MainScreen = {
  .Title = L"UiDemo - Main Menu",
  .Items = MainMenuItems,
  .ItemCount = sizeof(MainMenuItems)/sizeof(MainMenuItems[0]),
  .Selected = 0
};

EFI_STATUS EFIAPI UefiMain3(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable)
{
  EFI_STATUS Status;

  Status = GfxInit();
  if (EFI_ERROR(Status)) {
    // fallback to text mode: allow running the HelloBootServicesTest only
    Print(L"GOP not available (Status=%r). Running test in text mode or press Esc to quit.\n", Status);
    UINTN idx;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &idx);
    EFI_INPUT_KEY k;
    gST->ConIn->ReadKeyStroke(gST->ConIn, &k);
    if (k.ScanCode == SCAN_ESC) return EFI_SUCCESS;
    return Action_RunTest(ImageHandle, SystemTable, NULL);
  }

  gCreature = SpriteCreateProceduralCreature(4);
  if (gCreature == NULL) {
    Print(L"Warning: failed to create creature sprite\n");
  } else {
    MainMenuItems[1].Action = Action_CreatureDetail;
    MainMenuItems[1].Context = gCreature;
    MainMenuItems[2].Action = Action_SmokeDetail;
  }

  // push main screen
  UiPush(&MainScreen);

  // Create a timer event to drive animations in the main menu (preview)
  EFI_EVENT MenuTimer = NULL;
  Status = gBS->CreateEvent(EVT_TIMER, TPL_APPLICATION, NULL, NULL, &MenuTimer);
  if (!EFI_ERROR(Status)) {
    // set periodic timer: UI_FPS_MS milliseconds -> 100ns units
    gBS->SetTimer(MenuTimer, TimerPeriodic, (UINT64)UI_FPS_MS * 10000ULL);
  }

  // main loop
  while (UiTop() != NULL) {
    UI_SCREEN *cur = UiTop();

    // draw UI initially; each timer tick we will re-draw
    UiDrawCurrent(cur);

    // wait on timer + key
    EFI_EVENT WaitList[2];
    UINTN WaitCount = 0;
    if (MenuTimer != NULL) {
      WaitList[WaitCount++] = MenuTimer;
    }
    WaitList[WaitCount++] = gST->ConIn->WaitForKey;

    UINTN WaitIndex;
    Status = gBS->WaitForEvent(WaitCount, WaitList, &WaitIndex);
    if (EFI_ERROR(Status)) break;

    // Determine which event fired
    if (MenuTimer != NULL && WaitIndex == 0) {
      // timer tick -> advance preview frame and redraw
      static UINTN previewFrame = 0;
      previewFrame = (previewFrame + 1) % (gCreature ? gCreature->FrameCount : 1);

      // if current selection has creature context, draw live preview creature at corner
      if (cur->Selected < cur->ItemCount && cur->Items[cur->Selected].Context == gCreature && gCreature != NULL) {
        UINTN frame = previewFrame;
        UINT32 outW = gCreature->W;
        UINT32 outH = gCreature->H;
        UINT32 px = 30;
        UINT32 py = 60;
        PIXEL *frameBuf = &gCreature->Frames[frame * (gCreature->W * gCreature->H)];
        UINT8  *frameAlpha = &gCreature->Alpha[frame * (gCreature->W * gCreature->H)];
        GfxBlitBufferToVideo(frameBuf, outW, outH, px, py, frameAlpha);
      }
      // loop to wait for next event (do not consume key)
      continue;
    }

    // Otherwise it's a key event (last one in WaitList)
    EFI_INPUT_KEY Key;
    Status = gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
    if (EFI_ERROR(Status)) {
      // no key read, continue
      continue;
    }

    if (Key.ScanCode == SCAN_UP) {
      if (cur->Selected > 0) cur->Selected--;
    } else if (Key.ScanCode == SCAN_DOWN) {
      if (cur->Selected + 1 < cur->ItemCount) cur->Selected++;
    } else if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN) {
      UI_MenuItem *mi = &cur->Items[cur->Selected];
      if (mi->Action) mi->Action(ImageHandle, SystemTable, mi->Context);
      if (mi->SubScreen) UiPush(mi->SubScreen);
    } else if (Key.ScanCode == SCAN_ESC) {
      UiPop();
    }
  }

  // Stop and close menu timer
  if (MenuTimer != NULL) {
    gBS->SetTimer(MenuTimer, TimerPeriodic, 0);
    gBS->CloseEvent(MenuTimer);
  }

  if (gCreature) SpriteFree(gCreature);

  gST->ConOut->ClearScreen(gST->ConOut);
  Print(L"UiAppGui exiting.\n");
  return EFI_SUCCESS;
}
