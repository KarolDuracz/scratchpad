// helloworld_gop_font_with_pointer_images.c
// Build with EDK2 (add to INF and compile).
#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Protocol/GraphicsOutput.h>
#include <Protocol/SimplePointer.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/DevicePath.h>
#include <Protocol/DevicePathToText.h>
#include <Guid/FileInfo.h>
#include <Library/PrintLib.h>


// small helpers for printing like the original code's TerminalPrintf:
#define TerminalPrintf(...) Print(__VA_ARGS__)

// Path where BMP images will be searched
#define MYIMAGES_REL_PATH L"\\EFI\\Boot\\myPics"

// Forward
EFI_STATUS GetGop(EFI_GRAPHICS_OUTPUT_PROTOCOL **OutGop);

// -------------------- existing globals & font code --------------------
STATIC BOOLEAN gFontBgTransparent = TRUE;
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL gFontBgColor = { .Blue = 0, .Green = 0, .Red = 0, .Reserved = 0 };
STATIC UINTN gFontScale = 2; // default scale factor
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL gFontFgColor = { .Blue = 0x00, .Green = 0xFF, .Red = 0xFF, .Reserved = 0 };
STATIC UINT8 gFontAlpha = 255; // 0..255

typedef struct { CHAR8 ch; UINT8 rows[7]; } GLYPH;

STATIC GLYPH sGlyphs[] = {
  { 'H', { 0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001 } },
  { 'E', { 0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111 } },
  { 'L', { 0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111 } },
  { 'O', { 0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110 } },
  { 'W', { 0b10001,0b10001,0b10001,0b10101,0b10101,0b11011,0b10001 } },
  { 'R', { 0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001 } },
  { 'D', { 0b11110,0b10001,0b10001,0b10001,0b10001,0b10001,0b11110 } },
  { '!', { 0b00100,0b00100,0b00100,0b00100,0b00100,0b00000,0b00100 } },
  { ',', { 0b00000,0b00000,0b00000,0b00000,0b00000,0b00100,0b01000 } },
  { ' ', { 0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000 } },
};

STATIC CONST GLYPH* FindGlyph(CHAR16 wch) {
  CHAR8 ch = (CHAR8)wch;
  if (ch >= 'a' && ch <= 'z') ch = (CHAR8)(ch - 'a' + 'A');
  for (UINTN i = 0; i < ARRAY_SIZE(sGlyphs); ++i) {
    if (sGlyphs[i].ch == ch) return &sGlyphs[i];
  }
  return NULL;
}

STATIC EFI_STATUS ReadPixel(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN X, UINTN Y, EFI_GRAPHICS_OUTPUT_BLT_PIXEL *Out) {
  if (!Gop || !Out) return EFI_INVALID_PARAMETER;
  return Gop->Blt(Gop, Out, EfiBltVideoToBltBuffer, X, Y, 0, 0, 1, 1, sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
}
STATIC EFI_STATUS WritePixel(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN X, UINTN Y, CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL *In) {
  if (!Gop || !In) return EFI_INVALID_PARAMETER;
  return Gop->Blt(Gop, (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)In, EfiBltBufferToVideo, 0, 0, X, Y, 1, 1, sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
}

STATIC VOID BlendPixel(const EFI_GRAPHICS_OUTPUT_BLT_PIXEL *src, EFI_GRAPHICS_OUTPUT_BLT_PIXEL *dst, UINT8 alpha) {
  UINTN inv = 255 - alpha;
  dst->Red   = (UINT8)((src->Red   * (UINT32)alpha + dst->Red   * inv) / 255);
  dst->Green = (UINT8)((src->Green * (UINT32)alpha + dst->Green * inv) / 255);
  dst->Blue  = (UINT8)((src->Blue  * (UINT32)alpha + dst->Blue  * inv) / 255);
}

// ---------- Buffered character drawing ----------
STATIC EFI_STATUS DrawCharBuffered(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN X, UINTN Y, CONST GLYPH *g) {
  if (!Gop || !g) return EFI_INVALID_PARAMETER;
  UINTN scale = gFontScale;
  UINTN w = 5 * scale;
  UINTN h = 7 * scale;
  UINTN PixelSize = sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
  UINTN RowBytes = w * PixelSize;
  UINTN BufSize = RowBytes * h;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL *Buf = AllocatePool(BufSize);
  if (!Buf) return EFI_OUT_OF_RESOURCES;

  EFI_STATUS Status = Gop->Blt(Gop, Buf, EfiBltVideoToBltBuffer, X, Y, 0, 0, w, h, RowBytes);
  if (EFI_ERROR(Status)) { FreePool(Buf); return Status; }

  for (UINTN row = 0; row < 7; ++row) {
    UINT8 bits = g->rows[row];
    for (UINTN col = 0; col < 5; ++col) {
      BOOLEAN bitSet = (bits >> (4 - col)) & 1;
      for (UINTN sy = 0; sy < scale; ++sy) {
        UINTN py = row * scale + sy;
        for (UINTN sx = 0; sx < scale; ++sx) {
          UINTN px = col * scale + sx;
          EFI_GRAPHICS_OUTPUT_BLT_PIXEL *pPixel = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)((UINT8*)Buf + (py * RowBytes) + (px * PixelSize));
          if (bitSet) {
            EFI_GRAPHICS_OUTPUT_BLT_PIXEL dst = *pPixel;
            EFI_GRAPHICS_OUTPUT_BLT_PIXEL src = gFontFgColor;
            EFI_GRAPHICS_OUTPUT_BLT_PIXEL out = dst;
            BlendPixel(&src, &out, gFontAlpha);
            *pPixel = out;
          } else {
            if (!gFontBgTransparent) {
              *pPixel = gFontBgColor;
            }
          }
        }
      }
    }
  }

  Status = Gop->Blt(Gop, Buf, EfiBltBufferToVideo, 0, 0, X, Y, w, h, RowBytes);
  FreePool(Buf);
  return Status;
}
STATIC EFI_STATUS DrawCharAt(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN x, UINTN y, CHAR16 ch) {
  const GLYPH *g = FindGlyph(ch);
  if (!g) return EFI_NOT_FOUND;
  return DrawCharBuffered(Gop, x, y, g);
}
STATIC EFI_STATUS DrawStringAt(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN x, UINTN y, CONST CHAR16 *Str) {
  if (!Gop || !Str) return EFI_INVALID_PARAMETER;
  UINTN cursorX = x;
  UINTN charWidth = 5 * gFontScale;
  UINTN spacing = gFontScale;
  while (*Str) {
    EFI_STATUS Status = DrawCharAt(Gop, cursorX, y, *Str);
    if (EFI_ERROR(Status)) return Status;
    cursorX += charWidth + spacing;
    ++Str;
  }
  return EFI_SUCCESS;
}

// Helper: print one simple pointer Mode structure
STATIC VOID PrintPointerModeInfo(EFI_SIMPLE_POINTER_PROTOCOL *Sp) {
  if (Sp == NULL) { Print(L"  <NULL SimplePointer protocol>\n"); return; }
  EFI_SIMPLE_POINTER_MODE *Mode = Sp->Mode;
  if (Mode == NULL) {
    Print(L"  SimplePointer Mode: <NULL>\n");
    return;
  }
  Print(L"  Mode: ResolutionX=%lu, ResolutionY=%lu, ResolutionZ=%lu, LeftButton=%u, RightButton=%u\n",
        (unsigned long)Mode->ResolutionX,
        (unsigned long)Mode->ResolutionY,
        (unsigned long)Mode->ResolutionZ,
        Mode->LeftButton ? 1 : 0,
        Mode->RightButton ? 1 : 0);
  Print(L"  WaitForInput event: %p\n", Sp->WaitForInput);
}

// Helpers
STATIC UINTN HexStrToUint(CONST CHAR16 *Str) {
  UINTN val = 0;
  while (*Str) {
    CHAR16 c = *Str++;
    if (c == L' ' || c == L'\0') break;
    val <<= 4;
    if (c >= L'0' && c <= L'9') val |= (UINTN)(c - L'0');
    else if (c >= L'A' && c <= L'F') val |= (UINTN)(10 + c - L'A');
    else if (c >= L'a' && c <= L'f') val |= (UINTN)(10 + c - L'a');
    else break;
  }
  return val;
}
STATIC UINTN StrToUintn(CONST CHAR16 *Str) {
  if (Str == NULL) return 0;
  UINTN Val = 0; BOOLEAN Any = FALSE;
  while (*Str) {
    if (*Str >= L'0' && *Str <= L'9') { Any = TRUE; Val = Val * 10 + (UINTN)(*Str - L'0'); Str++; } else break;
  }
  return Any ? Val : 0;
}

// -------------------- Pointer (mouse) support --------------------
STATIC EFI_SIMPLE_POINTER_PROTOCOL *gSimplePointer = NULL;
STATIC BOOLEAN gPointerMode = FALSE;
STATIC INTN gCursorX = -1, gCursorY = -1;
STATIC INTN gSavedCursorX = -1, gSavedCursorY = -1;
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL *gSavedCursorBg = NULL;
STATIC UINTN gCursorW = 12, gCursorH = 16; // cursor rectangle size
STATIC CONST UINT16 gCursorBitmap[16] = {
  0b100000000000,0b110000000000,0b111000000000,0b111100000000,
  0b111110000000,0b111111000000,0b111111100000,0b111111110000,
  0b111111100000,0b111101100000,0b111001100000,0b110001100000,
  0b100001100000,0b000001100000,0b000001000000,0b000000000000
};

STATIC UINTN ClampU(UINTN v, UINTN lo, UINTN hi) { if (v < lo) return lo; if (v > hi) return hi; return v; }

STATIC VOID RestoreSavedCursor(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop) {
  if (!gSavedCursorBg || gSavedCursorX < 0 || gSavedCursorY < 0) return;
  UINTN PixelSize = sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
  UINTN RowBytes = gCursorW * PixelSize;
  (void)Gop->Blt(Gop, gSavedCursorBg, EfiBltBufferToVideo, 0, 0, (UINTN)gSavedCursorX, (UINTN)gSavedCursorY, gCursorW, gCursorH, RowBytes);
  FreePool(gSavedCursorBg);
  gSavedCursorBg = NULL;
  gSavedCursorX = gSavedCursorY = -1;
}

STATIC EFI_STATUS DrawCursorAt(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, INTN newX, INTN newY) {
  if (!Gop) return EFI_INVALID_PARAMETER;
  UINTN screenW = Gop->Mode->Info->HorizontalResolution;
  UINTN screenH = Gop->Mode->Info->VerticalResolution;
  if (newX < 0) newX = 0;
  if (newY < 0) newY = 0;
  if ((UINTN)newX > screenW - gCursorW) newX = (INTN)(screenW - gCursorW);
  if ((UINTN)newY > screenH - gCursorH) newY = (INTN)(screenH - gCursorH);

  if (gSavedCursorBg && (gSavedCursorX != newX || gSavedCursorY != newY)) {
    RestoreSavedCursor(Gop);
  }

  if (!gSavedCursorBg) {
    UINTN PixelSize = sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
    UINTN RowBytes = gCursorW * PixelSize;
    UINTN BufSize = RowBytes * gCursorH;
    gSavedCursorBg = AllocatePool(BufSize);
    if (!gSavedCursorBg) return EFI_OUT_OF_RESOURCES;
    EFI_STATUS Status = Gop->Blt(Gop, gSavedCursorBg, EfiBltVideoToBltBuffer, (UINTN)newX, (UINTN)newY, 0, 0, gCursorW, gCursorH, RowBytes);
    if (EFI_ERROR(Status)) { FreePool(gSavedCursorBg); gSavedCursorBg = NULL; return Status; }
    gSavedCursorX = newX; gSavedCursorY = newY;

    EFI_GRAPHICS_OUTPUT_BLT_PIXEL *drawBuf = AllocatePool(BufSize);
    if (!drawBuf) { RestoreSavedCursor(Gop); return EFI_OUT_OF_RESOURCES; }
    CopyMem(drawBuf, gSavedCursorBg, BufSize);

    EFI_GRAPHICS_OUTPUT_BLT_PIXEL curColor = { .Blue = 0x00, .Green = 0x00, .Red = 0x00, .Reserved = 0 };
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL outline = { .Blue = 0xFF, .Green = 0xFF, .Red = 0xFF, .Reserved = 0 };

    for (UINTN ry = 0; ry < gCursorH; ++ry) {
      UINT16 rowbits = (ry < 16) ? gCursorBitmap[ry] : 0;
      for (UINTN rx = 0; rx < gCursorW; ++rx) {
        BOOLEAN set = (rowbits & (1u << (gCursorW - 1 - rx))) != 0;
        if (set) {
          EFI_GRAPHICS_OUTPUT_BLT_PIXEL *p = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)((UINT8*)drawBuf + ry * RowBytes + rx * PixelSize);
          *p = curColor;
        }
      }
    }
    for (UINTN ry = 0; ry < gCursorH; ++ry) {
      UINT16 rowbits = (ry < 16) ? gCursorBitmap[ry] : 0;
      for (UINTN rx = 0; rx < gCursorW; ++rx) {
        BOOLEAN set = (rowbits & (1u << (gCursorW - 1 - rx))) != 0;
        if (set) {
          for (INTN ny = -1; ny <= 1; ++ny) for (INTN nx = -1; nx <= 1; ++nx) {
            INTN px = (INTN)rx + nx;
            INTN py = (INTN)ry + ny;
            if (px >= 0 && py >= 0 && (UINTN)px < gCursorW && (UINTN)py < gCursorH) {
              UINT16 nrow = (py < 16) ? gCursorBitmap[py] : 0;
              BOOLEAN nset = (nrow & (1u << (gCursorW - 1 - px))) != 0;
              if (!nset) {
                EFI_GRAPHICS_OUTPUT_BLT_PIXEL *p = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)((UINT8*)drawBuf + py * RowBytes + px * sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
                *p = outline;
              }
            }
          }
        }
      }
    }

    EFI_STATUS s = Gop->Blt(Gop, drawBuf, EfiBltBufferToVideo, 0, 0, (UINTN)newX, (UINTN)newY, gCursorW, gCursorH, RowBytes);
    FreePool(drawBuf);
    if (EFI_ERROR(s)) { RestoreSavedCursor(Gop); return s; }
  }

  gCursorX = newX; gCursorY = newY;
  return EFI_SUCCESS;
}

STATIC VOID StartPointerMode() {
  if (gPointerMode) return;
  EFI_STATUS Status = gBS->LocateProtocol(&gEfiSimplePointerProtocolGuid, NULL, (VOID**)&gSimplePointer);
  if (EFI_ERROR(Status) || gSimplePointer == NULL) {
    TerminalPrintf(L"pointer_mode: SimplePointer protocol not found: %r\n", Status);
    gSimplePointer = NULL;
    gPointerMode = FALSE;
    return;
  }
  gPointerMode = TRUE;
  TerminalPrintf(L"pointer_mode: enabled (SimplePointer detected)\n");
}

STATIC VOID StopPointerMode(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop) {
  if (!gPointerMode) return;
  RestoreSavedCursor(Gop);
  gPointerMode = FALSE;
  gSimplePointer = NULL;
  TerminalPrintf(L"pointer_mode: disabled\n");
}

STATIC VOID DrawPointerLog(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, INTN x, INTN y) {
  if (!Gop) return;
  CHAR16 buf[64];
  UnicodeSPrint(buf, sizeof(buf), L"X:%d Y:%d", x, y);
  UINTN textW = (5 * gFontScale) * StrLen(buf) + gFontScale * StrLen(buf);
  UINTN textH = 7 * gFontScale;
  UINTN sx = 8, sy = 8;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL bg = { .Blue = 0x33, .Green = 0x33, .Red = 0x33, .Reserved = 0 };
  Gop->Blt(Gop, &bg, EfiBltVideoFill, 0, 0, sx-2, sy-2, textW+4, textH+4, 0);
  BOOLEAN oldTransparent = gFontBgTransparent;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL oldBg = gFontBgColor;
  gFontBgTransparent = FALSE;
  gFontBgColor.Red = 0x33; gFontBgColor.Green = 0x33; gFontBgColor.Blue = 0x33;
  DrawStringAt(Gop, sx, sy, buf);
  gFontBgTransparent = oldTransparent;
  gFontBgColor = oldBg;
}

// -------------------- Image list & BMP loader --------------------
typedef struct {
  CHAR16 *Name;           // filename only (allocated)
  CHAR16 *FullPath;       // full path (allocated)
} IMAGE_ENTRY;

STATIC IMAGE_ENTRY *gImageList = NULL;
STATIC UINTN gImageCount = 0;

// Loaded image in memory (converted to EFI_GRAPHICS_OUTPUT_BLT_PIXEL)
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL *gLoadedImagePixels = NULL;
STATIC UINTN gLoadedImageWidth = 0;
STATIC UINTN gLoadedImageHeight = 0;
STATIC CHAR16 *gLoadedImageName = NULL;

STATIC VOID FreeImageList() {
  if (!gImageList) return;
  for (UINTN i = 0; i < gImageCount; ++i) {
    if (gImageList[i].Name) FreePool(gImageList[i].Name);
    if (gImageList[i].FullPath) FreePool(gImageList[i].FullPath);
  }
  FreePool(gImageList);
  gImageList = NULL;
  gImageCount = 0;
}

STATIC VOID FreeLoadedImage() {
  if (gLoadedImagePixels) { FreePool(gLoadedImagePixels); gLoadedImagePixels = NULL; }
  gLoadedImageWidth = gLoadedImageHeight = 0;
  if (gLoadedImageName) { FreePool(gLoadedImageName); gLoadedImageName = NULL; }
}

// Helper: join root path + relative. Root is volume root (we will open using the file protocol).
// But we will pass already formed MYIMAGES_REL_PATH to Open — no need to join here.
// Enumerate files in MYIMAGES_REL_PATH
STATIC EFI_STATUS EnumerateImages(VOID) {
  FreeImageList();

  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiSimpleFileSystemProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    if (Handles) FreePool(Handles);
    return EFI_NOT_FOUND;
  }

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *Sfsp = NULL;
    Status = gBS->HandleProtocol(Handles[i], &gEfiSimpleFileSystemProtocolGuid, (VOID**)&Sfsp);
    if (EFI_ERROR(Status) || Sfsp == NULL) continue;

    EFI_FILE_PROTOCOL *Root = NULL;
    Status = Sfsp->OpenVolume(Sfsp, &Root);
    if (EFI_ERROR(Status) || Root == NULL) continue;

    // Try to open the images directory
    EFI_FILE_PROTOCOL *Dir = NULL;
    Status = Root->Open(Root, &Dir, MYIMAGES_REL_PATH, EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status) || Dir == NULL) {
      // maybe the path doesn't exist on this volume
      if (Root) Root->Close(Root);
      continue;
    }

    // Iterate entries
    UINTN BufferSize = sizeof(EFI_FILE_INFO) + 512;
    UINT8 *InfoBuf = AllocatePool(BufferSize);
    if (!InfoBuf) { Dir->Close(Dir); Root->Close(Root); continue; }

    while (TRUE) {
      ZeroMem(InfoBuf, BufferSize);
      Status = Dir->Read(Dir, &BufferSize, InfoBuf);
      if (EFI_ERROR(Status) || BufferSize == 0) break;
      EFI_FILE_INFO *Fi = (EFI_FILE_INFO*)InfoBuf;
      // skip '.' and '..'
      if (Fi->FileName[0] == L'.') { BufferSize = sizeof(EFI_FILE_INFO) + 512; continue; }
      // Only files (not directories)
      if ((Fi->Attribute & EFI_FILE_DIRECTORY) == 0) {
        // Check extension .bmp (case-insensitive)
        CHAR16 *fn = Fi->FileName;
        UINTN len = StrLen(fn);
        if (len >= 4) {
          CHAR16 *ext = &fn[len - 4];
          if ((StrCmp(ext, L".bmp") == 0) || (StrCmp(ext, L".BMP") == 0)) {
            // Add to list
            IMAGE_ENTRY *newList = AllocatePool(sizeof(IMAGE_ENTRY) * (gImageCount + 1));
            if (!newList) continue;
            if (gImageList) CopyMem(newList, gImageList, sizeof(IMAGE_ENTRY) * gImageCount);
            // free old
            if (gImageList) FreePool(gImageList);
            gImageList = newList;
            gImageList[gImageCount].Name = AllocatePool((len + 1) * sizeof(CHAR16));
            StrCpyS(gImageList[gImageCount].Name, len + 1, fn);

            // Build full path: MYIMAGES_REL_PATH + L"\\" + filename
            UINTN fullLen = StrLen(MYIMAGES_REL_PATH) + 1 + len + 1;
            gImageList[gImageCount].FullPath = AllocatePool(fullLen * sizeof(CHAR16));
            UnicodeSPrint(gImageList[gImageCount].FullPath, fullLen * sizeof(CHAR16), L"%s\\%s", MYIMAGES_REL_PATH, fn);

            gImageCount++;
          }
        }
      }
      BufferSize = sizeof(EFI_FILE_INFO) + 512;
    }

    FreePool(InfoBuf);
    Dir->Close(Dir);
    Root->Close(Root);
  } // end handles

  if (Handles) FreePool(Handles);
  return (gImageCount > 0) ? EFI_SUCCESS : EFI_NOT_FOUND;
}

// Helper: open file, read all bytes into buffer and return pointer + size (caller must FreePool)
STATIC EFI_STATUS ReadFileFromVolume(CHAR16 *FullPath, VOID **OutBuf, UINTN *OutSize) {
  if (!FullPath || !OutBuf || !OutSize) return EFI_INVALID_PARAMETER;
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiSimpleFileSystemProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR(Status) || HandleCount == 0) { if (Handles) FreePool(Handles); return EFI_NOT_FOUND; }

  EFI_STATUS LastStatus = EFI_NOT_FOUND;

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *Sfsp = NULL;
    Status = gBS->HandleProtocol(Handles[i], &gEfiSimpleFileSystemProtocolGuid, (VOID**)&Sfsp);
    if (EFI_ERROR(Status) || Sfsp == NULL) continue;

    EFI_FILE_PROTOCOL *Root = NULL;
    Status = Sfsp->OpenVolume(Sfsp, &Root);
    if (EFI_ERROR(Status) || Root == NULL) continue;

    EFI_FILE_PROTOCOL *File = NULL;
    Status = Root->Open(Root, &File, FullPath, EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status) || File == NULL) { Root->Close(Root); LastStatus = Status; continue; }

    // get file size via GetInfo
    UINTN InfoSize = sizeof(EFI_FILE_INFO) + 512;
    EFI_FILE_INFO *Fi = AllocatePool(InfoSize);
    if (!Fi) { File->Close(File); Root->Close(Root); LastStatus = EFI_OUT_OF_RESOURCES; continue; }
    Status = File->GetInfo(File, &gEfiFileInfoGuid, &InfoSize, Fi);
    if (EFI_ERROR(Status)) { FreePool(Fi); File->Close(File); Root->Close(Root); LastStatus = Status; continue; }
    UINTN FileSize = (UINTN)Fi->FileSize;
    FreePool(Fi);

    VOID *Buf = AllocatePool(FileSize);
    if (!Buf) { File->Close(File); Root->Close(Root); LastStatus = EFI_OUT_OF_RESOURCES; continue; }

    UINTN ReadSize = FileSize;
    Status = File->Read(File, &ReadSize, Buf);
    if (EFI_ERROR(Status) || ReadSize != FileSize) { FreePool(Buf); File->Close(File); Root->Close(Root); LastStatus = Status; continue; }

    // success
    *OutBuf = Buf;
    *OutSize = FileSize;
    File->Close(File);
    Root->Close(Root);
    if (Handles) FreePool(Handles);
    return EFI_SUCCESS;
  }

  if (Handles) FreePool(Handles);
  return LastStatus;
}

// BMP header parsing (minimal): supports BITMAPFILEHEADER (14), BITMAPINFOHEADER (40),
// uncompressed RGB (biCompression == 0) with 24bpp or 32bpp
#pragma pack(push,1)
typedef struct {
  UINT16 bfType;      // 'BM' = 0x4D42
  UINT32 bfSize;
  UINT16 bfReserved1;
  UINT16 bfReserved2;
  UINT32 bfOffBits;
} BMP_FILE_HDR;

typedef struct {
  UINT32 biSize;      // should be 40
  INT32  biWidth;
  INT32  biHeight;
  UINT16 biPlanes;
  UINT16 biBitCount;
  UINT32 biCompression;
  UINT32 biSizeImage;
  INT32  biXPelsPerMeter;
  INT32  biYPelsPerMeter;
  UINT32 biClrUsed;
  UINT32 biClrImportant;
} BMP_INFO_HDR;
#pragma pack(pop)

// Load BMP from memory buffer (raw bytes) and convert to EFI_GRAPHICS_OUTPUT_BLT_PIXEL* (allocated)
// Supports 24bpp (BGR) and 32bpp (BGRA) uncompressed. Handles bottom-up BMP (positive height) and top-down (negative height).
STATIC EFI_STATUS ConvertBmpToBlt(VOID *BmpBuf, UINTN BmpSize,
                                  EFI_GRAPHICS_OUTPUT_BLT_PIXEL **OutPixels,
                                  UINTN *OutW, UINTN *OutH,
                                  UINTN *OutRowPixelBytes,
                                  CHAR16 **OutInfo)
{
  if (!BmpBuf || !OutPixels || !OutW || !OutH) return EFI_INVALID_PARAMETER;
  UINT8 *buf = (UINT8*)BmpBuf;
  if (BmpSize < sizeof(BMP_FILE_HDR) + sizeof(BMP_INFO_HDR)) return EFI_COMPROMISED_DATA;

  BMP_FILE_HDR *fh = (BMP_FILE_HDR*)buf;
  if (fh->bfType != 0x4D42) return EFI_COMPROMISED_DATA; // 'BM'

  BMP_INFO_HDR *ih = (BMP_INFO_HDR*)(buf + sizeof(BMP_FILE_HDR));
  if (ih->biSize < 40) return EFI_COMPROMISED_DATA;

  INT32 width = ih->biWidth;
  INT32 height = ih->biHeight;
  UINT16 bpp = ih->biBitCount;
  UINT32 comp = ih->biCompression;

  // Only support uncompressed
  if (comp != 0) return EFI_UNSUPPORTED;

  if (width <= 0 || height == 0) return EFI_COMPROMISED_DATA;
  BOOLEAN bottomUp = TRUE;
  if (height < 0) { bottomUp = FALSE; height = -height; }

  // Only 24 or 32
  if (bpp != 24 && bpp != 32) return EFI_UNSUPPORTED;

  // data pointer
  UINT32 off = fh->bfOffBits;
  if (off >= BmpSize) return EFI_COMPROMISED_DATA;
  UINT8 *pixelData = buf + off;
  UINTN bytesAvailable = BmpSize - off;

  UINTN rowBytesSrc; // bytes per bitmap row in file (including padding)
  if (bpp == 24) {
    // rows are aligned to 4 bytes
    UINTN unpadded = (UINTN)width * 3;
    rowBytesSrc = (unpadded + 3) & ~3;
  } else {
    // 32bpp: no padding typically
    rowBytesSrc = (UINTN)width * 4;
  }

  // check available size
  if (bytesAvailable < rowBytesSrc * (UINTN)height) return EFI_COMPROMISED_DATA;

  // allocate output pixels
  UINTN outW = (UINTN)width;
  UINTN outH = (UINTN)height;
  UINTN PixelBytes = sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
  UINTN RowPixelBytes = outW * PixelBytes;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL *pixels = AllocateZeroPool(RowPixelBytes * outH);
  if (!pixels) return EFI_OUT_OF_RESOURCES;

  // Convert
  for (UINTN row = 0; row < outH; ++row) {
    UINTN srcRow = bottomUp ? (outH - 1 - row) : row;
    UINT8 *src = pixelData + srcRow * rowBytesSrc;
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL *dst = pixels + row * outW;
    for (UINTN col = 0; col < outW; ++col) {
      if (bpp == 24) {
        UINT8 b = src[col * 3 + 0];
        UINT8 g = src[col * 3 + 1];
        UINT8 r = src[col * 3 + 2];
        dst[col].Blue = b; dst[col].Green = g; dst[col].Red = r; dst[col].Reserved = 0;
      } else { // 32
        UINT8 b = src[col * 4 + 0];
        UINT8 g = src[col * 4 + 1];
        UINT8 r = src[col * 4 + 2];
        UINT8 a = src[col * 4 + 3];
        dst[col].Blue = b; dst[col].Green = g; dst[col].Red = r; dst[col].Reserved = a;
      }
    }
  }

  // Build info string
  UINTN infoSize = 256;
  CHAR16 *info = AllocatePool(infoSize * sizeof(CHAR16));
  if (info) {
    UnicodeSPrint(info, infoSize * sizeof(CHAR16),
                  L"BMP: %ux%u, %u bpp, bottomUp=%u, offset=%u, imageSize=%u",
                  outW, outH, bpp, bottomUp ? 1 : 0, off, (UINT32)ih->biSizeImage);
  }

  *OutPixels = pixels;
  *OutW = outW;
  *OutH = outH;
  if (OutRowPixelBytes) *OutRowPixelBytes = RowPixelBytes;
  if (OutInfo) *OutInfo = info;
  return EFI_SUCCESS;
}

// Load BMP file by list index: reads, parses and stores into gLoadedImagePixels
STATIC EFI_STATUS LoadImageByIndex(UINTN Index) {
  if (Index >= gImageCount) return EFI_INVALID_PARAMETER;
  FreeLoadedImage();
  VOID *Buf = NULL;
  UINTN BufSize = 0;
  EFI_STATUS Status = ReadFileFromVolume(gImageList[Index].FullPath, &Buf, &BufSize);
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"images load: failed to read '%s': %r\n", gImageList[Index].FullPath, Status);
    return Status;
  }
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL *pixels = NULL;
  UINTN w=0,h=0,rowBytes=0;
  CHAR16 *info = NULL;
  Status = ConvertBmpToBlt(Buf, BufSize, &pixels, &w, &h, &rowBytes, &info);
  FreePool(Buf);
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"images load: failed to parse BMP '%s': %r\n", gImageList[Index].FullPath, Status);
    return Status;
  }
  gLoadedImagePixels = pixels;
  gLoadedImageWidth = w;
  gLoadedImageHeight = h;
  gLoadedImageName = AllocateCopyPool((StrLen(gImageList[Index].Name)+1)*sizeof(CHAR16), gImageList[Index].Name);
  TerminalPrintf(L"Loaded image[%u] '%s' : %s\n", Index, gImageList[Index].Name, info ? info : L"(info N/A)");
  if (info) FreePool(info);
  return EFI_SUCCESS;
}

// Draw loaded image at X,Y (clip if outside)
STATIC EFI_STATUS DrawLoadedImageAt(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN X, UINTN Y) {
  if (!Gop) return EFI_INVALID_PARAMETER;
  if (!gLoadedImagePixels) { TerminalPrintf(L"add_image: no image loaded (use 'gop images load N')\n"); return EFI_NOT_READY; }

  UINTN screenW = Gop->Mode->Info->HorizontalResolution;
  UINTN screenH = Gop->Mode->Info->VerticalResolution;

  if (X >= screenW || Y >= screenH) { TerminalPrintf(L"add_image: coords out of screen\n"); return EFI_INVALID_PARAMETER; }

  // clip width/height to screen
  UINTN drawW = gLoadedImageWidth;
  UINTN drawH = gLoadedImageHeight;
  if (X + drawW > screenW) drawW = screenW - X;
  if (Y + drawH > screenH) drawH = screenH - Y;

  // If we can blit the entire loaded image or a partial rectangle,
  // use Gop->Blt with EfiBltBufferToVideo. We must provide RowPitch bytes:
  UINTN RowPixels = gLoadedImageWidth;
  UINTN RowBytes = RowPixels * sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);

  // If full width and height -> copy whole; otherwise copy row-by-row
  if (drawW == gLoadedImageWidth && drawH == gLoadedImageHeight) {
    EFI_STATUS s = Gop->Blt(Gop, gLoadedImagePixels, EfiBltBufferToVideo, 0, 0, X, Y, drawW, drawH, RowBytes);
    if (EFI_ERROR(s)) { TerminalPrintf(L"add_image: BLT failed: %r\n", s); return s; }
  } else {
    // Blit row by row
    for (UINTN row = 0; row < drawH; ++row) {
      EFI_GRAPHICS_OUTPUT_BLT_PIXEL *rowPtr = gLoadedImagePixels + row * gLoadedImageWidth;
      EFI_STATUS s = Gop->Blt(Gop, rowPtr, EfiBltBufferToVideo, 0, 0, X, Y + row, drawW, 1, RowBytes);
      if (EFI_ERROR(s)) { TerminalPrintf(L"add_image: BLT row failed: %r\n", s); return s; }
    }
  }
  TerminalPrintf(L"add_image: drawn '%s' at %u,%u size %ux%u\n", gLoadedImageName ? gLoadedImageName : L"(?)", X, Y, drawW, drawH);
  return EFI_SUCCESS;
}

// -------------------- DoGopCommand updated with pointer_mode & images --------------------
STATIC VOID DoGopCommand(CONST CHAR16 *Arg) {
  if (Arg == NULL) {
    TerminalPrintf(L"gop: missing argument (list|set|bg|fontbg|fontfg|fontsize|draw|blendtest|pointer_mode|images|add_image)\n");
    return;
  }

  CHAR16 Work[1024];
  StrCpyS(Work, ARRAY_SIZE(Work), Arg);
  CHAR16 *Sub = Work;
  CHAR16 *Rest = StrStr(Sub, L" ");
  if (Rest) { *Rest = L'\0'; Rest++; while (*Rest == L' ') Rest++; }

  // pointer_mode handling (on|off|device)
  if (StrCmp(Sub, L"pointer_mode") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop pointer_mode: missing arg (on|off|device)\n"); return; }
    if (StrCmp(Rest, L"on") == 0) {
      StartPointerMode();
      return;
    } else if (StrCmp(Rest, L"off") == 0) {
      EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
      if (!EFI_ERROR(GetGop(&Gop))) StopPointerMode(Gop);
      else StopPointerMode(NULL);
      return;
    } else if (StrCmp(Rest, L"device") == 0) {
      // Enumerate simple pointer handles and print info, then monitor the first device
      EFI_STATUS Status;
      EFI_HANDLE *Handles = NULL;
      UINTN HandleCount = 0;
      Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiSimplePointerProtocolGuid, NULL, &HandleCount, &Handles);
      if (EFI_ERROR(Status) || HandleCount == 0) {
        TerminalPrintf(L"gop pointer_mode device: no SimplePointer handles found (%r)\n", Status);
        if (Handles) FreePool(Handles);
        return;
      }
      TerminalPrintf(L"gop pointer_mode device: found %u SimplePointer handle(s)\n", HandleCount);

      EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpText = NULL;
      Status = gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID**)&DpText);
      if (EFI_ERROR(Status)) DpText = NULL;

      for (UINTN i = 0; i < HandleCount; ++i) {
        EFI_HANDLE h = Handles[i];
        TerminalPrintf(L"Handle[%u] = %p\n", i, h);
        EFI_SIMPLE_POINTER_PROTOCOL *Sp = NULL;
        Status = gBS->HandleProtocol(h, &gEfiSimplePointerProtocolGuid, (VOID**)&Sp);
        if (EFI_ERROR(Status) || Sp == NULL) {
          TerminalPrintf(L"  Could not get SimplePointer protocol: %r\n", Status);
        } else {
          PrintPointerModeInfo(Sp);
        }
        EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
        Status = gBS->HandleProtocol(h, &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
        if (!EFI_ERROR(Status) && DevPath != NULL && DpText != NULL) {
          CHAR16 *DpStr = DpText->ConvertDevicePathToText(DevPath, TRUE, TRUE);
          if (DpStr) {
            TerminalPrintf(L"  DevicePath: %s\n", DpStr);
            FreePool(DpStr);
          } else {
            TerminalPrintf(L"  DevicePath: (ConvertDevicePathToText returned NULL)\n");
          }
        } else {
          TerminalPrintf(L"  DevicePath: not available (HandleProtocol returned %r or converter not found)\n", Status);
        }
      }

      // Monitor first device (live)
      {
        EFI_SIMPLE_POINTER_PROTOCOL *MonitorSp = NULL;
        EFI_HANDLE monitorHandle = Handles[0];
        Status = gBS->HandleProtocol(monitorHandle, &gEfiSimplePointerProtocolGuid, (VOID**)&MonitorSp);
        if (EFI_ERROR(Status) || MonitorSp == NULL) {
          TerminalPrintf(L"gop pointer_mode device: cannot open first SimplePointer for monitoring: %r\n", Status);
          FreePool(Handles);
          return;
        }
        TerminalPrintf(L"\nMonitoring SimplePointer on handle %p. Move the device to see RelativeMovement values.\n", monitorHandle);
        TerminalPrintf(L"Press any key to stop monitoring.\n");

        INT64 posX, posY;
        EFI_GRAPHICS_OUTPUT_PROTOCOL *GopLocal = NULL;
        if (EFI_ERROR(GetGop(&GopLocal)) || GopLocal == NULL) {
          posX = 0; posY = 0;
        } else {
          posX = (INT64)(GopLocal->Mode->Info->HorizontalResolution) / 2;
          posY = (INT64)(GopLocal->Mode->Info->VerticalResolution) / 2;
        }

        EFI_SIMPLE_POINTER_STATE State;
        EFI_STATUS getst;
        BOOLEAN useWaitForInput = (MonitorSp->WaitForInput != NULL);
        while (TRUE) {
          if (useWaitForInput) {
            EFI_EVENT ev[2];
            ev[0] = gST->ConIn->WaitForKey;
            ev[1] = MonitorSp->WaitForInput;
            UINTN idx;
            Status = gBS->WaitForEvent(2, ev, &idx);
            if (EFI_ERROR(Status)) break;
            if (idx == 0) {
              EFI_INPUT_KEY K;
              while (gST->ConIn->ReadKeyStroke(gST->ConIn, &K) == EFI_SUCCESS) { /* consume */ }
              break;
            }
            getst = MonitorSp->GetState(MonitorSp, &State);
            if (!EFI_ERROR(getst)) {
              posX += (INT64)State.RelativeMovementX;
              posY += (INT64)State.RelativeMovementY;
              TerminalPrintf(L"Pointer event: RelX=%d RelY=%d RelZ=%d Left=%u Right=%u  PosX=%ld PosY=%ld\n",
                             (INT32)State.RelativeMovementX, (INT32)State.RelativeMovementY, (INT32)State.RelativeMovementZ,
                             State.LeftButton ? 1 : 0, State.RightButton ? 1 : 0,
                             (long)posX, (long)posY);
            } else if (getst == EFI_NOT_READY) {
            } else {
              TerminalPrintf(L"  GetState returned %r\n", getst);
            }
            continue;
          }

          // polling fallback
          EFI_INPUT_KEY K;
          if (gST->ConIn->ReadKeyStroke(gST->ConIn, &K) == EFI_SUCCESS) {
            while (gST->ConIn->ReadKeyStroke(gST->ConIn, &K) == EFI_SUCCESS) { }
            break;
          }
          getst = MonitorSp->GetState(MonitorSp, &State);
          if (!EFI_ERROR(getst)) {
            posX += (INT64)State.RelativeMovementX;
            posY += (INT64)State.RelativeMovementY;
            TerminalPrintf(L"[poll] Pointer: RelX=%d RelY=%d Left=%u Right=%u  PosX=%ld PosY=%ld\n",
                           (INT32)State.RelativeMovementX, (INT32)State.RelativeMovementY,
                           State.LeftButton ? 1 : 0, State.RightButton ? 1 : 0,
                           (long)posX, (long)posY);
          } else if (getst != EFI_NOT_READY) {
            TerminalPrintf(L"[poll] GetState returned %r\n", getst);
          }
          gBS->Stall(100000);
        }

        TerminalPrintf(L"Monitoring ended (key pressed or error). Returning to prompt.\n");
      }

      FreePool(Handles);
      return;
    } else {
      TerminalPrintf(L"gop pointer_mode: unknown arg '%s' (use on|off|device)\n", Rest);
      return;
    }
  }

  // images commands: "images list", "images load N"
  if (StrCmp(Sub, L"images") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop images: missing subcommand (list|load)\n"); return; }
    CHAR16 *sub2 = Rest;
    CHAR16 *rest2 = StrStr(sub2, L" ");
    if (rest2) { *rest2 = L'\0'; rest2++; while (*rest2 == L' ') rest2++; }
    if (StrCmp(sub2, L"list") == 0) {
      EFI_STATUS s = EnumerateImages();
      if (EFI_ERROR(s)) { TerminalPrintf(L"gop images list: none found (%r)\n", s); return; }
      TerminalPrintf(L"gop images list: %u image(s) in %s\n", gImageCount, MYIMAGES_REL_PATH);
      for (UINTN i = 0; i < gImageCount; ++i) {
        TerminalPrintf(L"  [%u] %s\n", i, gImageList[i].Name);
      }
      return;
    } else if (StrCmp(sub2, L"load") == 0) {
      if (rest2 == NULL) { TerminalPrintf(L"gop images load: missing index\n"); return; }
      UINTN idx = StrToUintn(rest2);
      EFI_STATUS s = EnumerateImages();
      if (EFI_ERROR(s)) { TerminalPrintf(L"gop images load: enumeration failed (%r)\n", s); return; }
      if (idx >= gImageCount) { TerminalPrintf(L"gop images load: invalid index %u\n", idx); return; }
      s = LoadImageByIndex(idx);
      if (EFI_ERROR(s)) { TerminalPrintf(L"gop images load: failed %r\n", s); return; }
      TerminalPrintf(L"gop images load: loaded index %u '%s' size %ux%u\n", idx, gImageList[idx].Name, gLoadedImageWidth, gLoadedImageHeight);
      return;
    } else {
      TerminalPrintf(L"gop images: unknown subcommand '%s'\n", sub2); return;
    }
  }

  // add_image X Y -> blit currently loaded image at X Y
  if (StrCmp(Sub, L"add_image") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop add_image: missing args (X Y)\n"); return; }
    CHAR16 *p = Rest;
    CHAR16 *p2 = StrStr(p, L" ");
    if (!p2) { TerminalPrintf(L"gop add_image: missing Y\n"); return; }
    *p2 = L'\0'; p2++; while (*p2 == L' ') p2++;
    UINTN x = StrToUintn(p);
    UINTN y = StrToUintn(p2);
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop add_image: GOP not available: %r\n", Status); return; }
    Status = DrawLoadedImageAt(Gop, x, y);
    if (EFI_ERROR(Status)) TerminalPrintf(L"gop add_image: failed: %r\n", Status);
    return;
  }

  // --- existing other commands (list,set,bg,fontbg,fontfg,fontsize,draw,blendtest) ---
  if (StrCmp(Sub, L"list") == 0) {
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop list: GOP not available: %r\n", Status); return; }
    TerminalPrintf(L"Mode Count: %u, CurrentMode: %u\n", Gop->Mode->MaxMode, Gop->Mode->Mode);
    for (UINT32 i = 0; i < Gop->Mode->MaxMode; ++i) {
      EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *Info;
      UINTN SizeOfInfo;
      Status = Gop->QueryMode(Gop, i, &SizeOfInfo, &Info);
      if (EFI_ERROR(Status)) continue;
      TerminalPrintf(L"  [%u] %ux%u, PixelFormat %u, PixelsPerScanLine %u\n",
                     i, Info->HorizontalResolution, Info->VerticalResolution, Info->PixelFormat, Info->PixelsPerScanLine);
    }
    return;
  } else if (StrCmp(Sub, L"set") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop set: missing mode index\n"); return; }
    UINTN idx = StrToUintn(Rest);
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop set: GOP not available: %r\n", Status); return; }
    if (idx >= Gop->Mode->MaxMode) { TerminalPrintf(L"gop set: invalid mode index %u\n", idx); return; }
    Status = Gop->SetMode(Gop, (UINT32)idx);
    if (EFI_ERROR(Status)) TerminalPrintf(L"gop set: SetMode failed: %r\n", Status); else TerminalPrintf(L"gop set: mode changed to %u\n", idx);
    return;
  } else if (StrCmp(Sub, L"bg") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop bg: missing hex color (RRGGBB)\n"); return; }
    UINTN val = HexStrToUint(Rest);
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop bg: GOP not available: %r\n", Status); return; }
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL Color;
    Color.Red = (UINT8)((val >> 16) & 0xFF);
    Color.Green = (UINT8)((val >> 8) & 0xFF);
    Color.Blue = (UINT8)(val & 0xFF);
    Color.Reserved = 0;
    Status = Gop->Blt(Gop, &Color, EfiBltVideoFill, 0, 0, 0, 0, Gop->Mode->Info->HorizontalResolution, Gop->Mode->Info->VerticalResolution, 0);
    if (EFI_ERROR(Status)) TerminalPrintf(L"gop bg: BLT failed: %r\n", Status); else TerminalPrintf(L"gop bg: background set to #%06x\n", val);
    return;
  } else if (StrCmp(Sub, L"fontbg") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop fontbg: missing argument (transparent|opaque|color RRGGBB)\n"); return; }
    CHAR16 *arg1 = Rest;
    CHAR16 *arg2 = StrStr(arg1, L" ");
    if (arg2) { *arg2 = L'\0'; arg2++; while (*arg2 == L' ') arg2++; }
    if (StrCmp(arg1, L"transparent") == 0) { gFontBgTransparent = TRUE; TerminalPrintf(L"gop fontbg: transparent enabled\n"); return; }
    else if (StrCmp(arg1, L"opaque") == 0) { gFontBgTransparent = FALSE; TerminalPrintf(L"gop fontbg: transparent disabled (opaque)\n"); return; }
    else if (StrCmp(arg1, L"color") == 0) {
      if (arg2 == NULL) { TerminalPrintf(L"gop fontbg: missing hex color after 'color'\n"); return; }
      UINTN val = HexStrToUint(arg2);
      gFontBgColor.Red = (UINT8)((val >> 16) & 0xFF); gFontBgColor.Green = (UINT8)((val >> 8) & 0xFF); gFontBgColor.Blue = (UINT8)(val & 0xFF);
      gFontBgColor.Reserved = 0; TerminalPrintf(L"gop fontbg: background color set to #%06x\n", val); return;
    } else { TerminalPrintf(L"gop fontbg: unknown option '%s'\n", arg1); return; }
  } else if (StrCmp(Sub, L"fontfg") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop fontfg: missing argument (color RRGGBB|alpha N)\n"); return; }
    CHAR16 *arg1 = Rest;
    CHAR16 *arg2 = StrStr(arg1, L" ");
    if (arg2) { *arg2 = L'\0'; arg2++; while (*arg2 == L' ') arg2++; }
    if (StrCmp(arg1, L"color") == 0) {
      if (arg2 == NULL) { TerminalPrintf(L"gop fontfg: missing hex color after 'color'\n"); return; }
      UINTN val = HexStrToUint(arg2);
      gFontFgColor.Red = (UINT8)((val >> 16) & 0xFF); gFontFgColor.Green = (UINT8)((val >> 8) & 0xFF); gFontFgColor.Blue = (UINT8)(val & 0xFF);
      gFontFgColor.Reserved = 0; TerminalPrintf(L"gop fontfg: foreground color set to #%06x\n", val); return;
    } else if (StrCmp(arg1, L"alpha") == 0) {
      if (arg2 == NULL) { TerminalPrintf(L"gop fontfg: missing alpha value after 'alpha'\n"); return; }
      UINTN a = StrToUintn(arg2); if (a > 255) a = 255; gFontAlpha = (UINT8)a; TerminalPrintf(L"gop fontfg: alpha set to %u\n", gFontAlpha); return;
    } else { TerminalPrintf(L"gop fontfg: unknown option '%s'\n", arg1); return; }
  } else if (StrCmp(Sub, L"fontsize") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop fontsize: missing scale factor\n"); return; }
    UINTN s = StrToUintn(Rest); if (s < 1) s = 1; gFontScale = s; TerminalPrintf(L"gop fontsize: scale set to %u\n", gFontScale); return;
  } else if (StrCmp(Sub, L"draw") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop draw: missing args\n"); return; }
    CHAR16 *p = Rest;
    CHAR16 *p2 = StrStr(p, L" ");
    if (!p2) { TerminalPrintf(L"gop draw: missing Y and text\n"); return; }
    *p2 = L'\0'; p2++; while (*p2 == L' ') p2++;
    UINTN x = StrToUintn(p);
    CHAR16 *p3 = StrStr(p2, L" ");
    if (!p3) { TerminalPrintf(L"gop draw: missing text\n"); return; }
    *p3 = L'\0'; p3++; while (*p3 == L' ') p3++;
    UINTN y = StrToUintn(p2);
    CHAR16 *txt = p3;
    if (*txt == L'"') { txt++; CHAR16 *q = StrStr(txt, L"\""); if (q) *q = L'\0'; }
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop draw: GOP not available: %r\n", Status); return; }
    Status = DrawStringAt(Gop, x, y, txt);
    if (EFI_ERROR(Status)) TerminalPrintf(L"gop draw: draw failed: %r\n", Status); else TerminalPrintf(L"gop draw: drew '%s' at %u,%u\n", txt, x, y);
    return;
  } else if (StrCmp(Sub, L"blendtest") == 0) {
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop blendtest: GOP not available: %r\n", Status); return; }
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL blue = { .Blue = 0xFF, .Green = 0x00, .Red = 0x00, .Reserved = 0 };
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL red  = { .Blue = 0x00, .Green = 0x00, .Red = 0xFF, .Reserved = 0 };
    UINTN x0 = 50, y0 = 50;
    for (UINTN y = y0; y < y0 + 80; ++y) for (UINTN x = x0; x < x0 + 160; ++x) WritePixel(Gop, x, y, &blue);
    for (UINTN y = y0 + 20; y < y0 + 100; ++y) for (UINTN x = x0 + 40; x < x0 + 200; ++x) {
      EFI_GRAPHICS_OUTPUT_BLT_PIXEL dst;
      if (EFI_ERROR(ReadPixel(Gop, x, y, &dst))) continue;
      EFI_GRAPHICS_OUTPUT_BLT_PIXEL out = dst;
      BlendPixel(&red, &out, 128);
      WritePixel(Gop, x, y, &out);
    }
    TerminalPrintf(L"gop blendtest: finished\n");
    return;
  }

  TerminalPrintf(L"gop: unknown subcommand '%s'\n", Sub);
}

// -------------------- GetGop --------------------
EFI_STATUS GetGop(EFI_GRAPHICS_OUTPUT_PROTOCOL **OutGop) {
  return gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID**)OutGop);
}

// -------------------- Entry point & interactive loop (with pointer support) --------------------
EFI_STATUS EFIAPI UefiMain(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable) {
  EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
  EFI_STATUS Status = GetGop(&Gop);
  if (EFI_ERROR(Status)) {
    Print(L"helloworld: GOP not available: %r\n", Status);
  } else {
    Print(L"helloworld: GOP mode: %ux%u (Mode %u)\n",
          Gop->Mode->Info->HorizontalResolution, Gop->Mode->Info->VerticalResolution, Gop->Mode->Mode);
  }

  Print(L"Simple interactive demo (examples):\n");
  Print(L"  gop list\n");
  Print(L"  gop set 1\n");
  Print(L"  gop bg FF00FF\n");
  Print(L"  gop fontbg transparent\n");
  Print(L"  gop fontsize 3\n");
  Print(L"  gop fontfg color FFFF00\n");
  Print(L"  gop draw 50 50 \"HELLO, WORLD!\"\n");
  Print(L"  gop blendtest\n");
  Print(L"  gop pointer_mode on\n");
  Print(L"  gop pointer_mode device\n");
  Print(L"  gop images list\n");
  Print(L"  gop images load 0\n");
  Print(L"  gop add_image 100 100\n\n");

  // Demo initial draw
  gFontBgTransparent = TRUE;
  gFontScale = 3;
  gFontFgColor.Red = 0xFF; gFontFgColor.Green = 0xFF; gFontFgColor.Blue = 0x00;
  gFontAlpha = 255;
  if (!EFI_ERROR(GetGop(&Gop))) {
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL bg = { .Blue = 0x33, .Green = 0x33, .Red = 0x33, .Reserved = 0 };
    Gop->Blt(Gop, &bg, EfiBltVideoFill, 0,0,0,0,Gop->Mode->Info->HorizontalResolution,Gop->Mode->Info->VerticalResolution,0);
    DrawStringAt(Gop, 60, 40, L"HELLO, WORLD!");
  }

  // Example commands
  DoGopCommand(L"list");
  DoGopCommand(L"fontsize 2");
  DoGopCommand(L"fontbg color FF0000");
  DoGopCommand(L"fontbg opaque");
  DoGopCommand(L"draw 60 120 \"HELLO, WORLD!\"");

  // Interactive command loop with pointer event integration
  {
    BOOLEAN Done = FALSE;
    CHAR16 Line[1024];
    UINTN Len;

    Print(L"\nEntering interactive mode. Type 'exit' to quit.\n");
    while (!Done) {
      EFI_EVENT Events[2];
      UINTN EventCount = 0;
      Events[EventCount++] = gST->ConIn->WaitForKey;
      if (gPointerMode && gSimplePointer && gSimplePointer->WaitForInput) {
        Events[EventCount++] = gSimplePointer->WaitForInput;
      }

      Print(L"> ");
      Len = 0;
      Line[0] = L'\0';

      while (TRUE) {
        UINTN Index;
        EFI_STATUS WaitStatus = gBS->WaitForEvent(EventCount, Events, &Index);
        if (EFI_ERROR(WaitStatus)) break;

        if (gPointerMode && EventCount == 2 && Index == 1) {
          EFI_SIMPLE_POINTER_STATE State;
          if (gSimplePointer && gSimplePointer->GetState && !EFI_ERROR(gSimplePointer->GetState(gSimplePointer, &State))) {
            INTN dx = (INTN)State.RelativeMovementX;
            INTN dy = (INTN)State.RelativeMovementY;
            if (dx != 0 || dy != 0 || gCursorX < 0) {
              INTN newX = (gCursorX < 0) ? (INTN)(Gop->Mode->Info->HorizontalResolution / 2) : gCursorX + dx;
              INTN newY = (gCursorY < 0) ? (INTN)(Gop->Mode->Info->VerticalResolution / 2) : gCursorY + dy;
              if ((UINTN)newX >= Gop->Mode->Info->HorizontalResolution) newX = (INTN)(Gop->Mode->Info->HorizontalResolution - 1);
              if ((UINTN)newY >= Gop->Mode->Info->VerticalResolution) newY = (INTN)(Gop->Mode->Info->VerticalResolution - 1);
              DrawCursorAt(Gop, newX, newY);
              DrawPointerLog(Gop, newX, newY);
            }
          }
          continue;
        }

        EFI_INPUT_KEY Key;
        if (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_SUCCESS) {
          if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN || Key.UnicodeChar == CHAR_LINEFEED) { Print(L"\r\n"); break; }
          if (Key.UnicodeChar == CHAR_BACKSPACE) { if (Len > 0) { Len--; Line[Len] = L'\0'; Print(L"\b \b"); } continue; }
          if (Key.UnicodeChar == 0) { continue; }
          if (Len + 1 < ARRAY_SIZE(Line)) { Line[Len++] = Key.UnicodeChar; Line[Len] = L'\0'; Print(L"%c", Key.UnicodeChar); }
        }
      } // end inner wait loop

      CHAR16 *cmd = Line;
      while (*cmd == L' ') cmd++;

      if (StrLen(cmd) == 0) continue;
      if (StrCmp(cmd, L"exit") == 0) { Done = TRUE; continue; }

      if (StrnCmp(cmd, L"gop", 3) == 0) {
        if (cmd[3] == L' ' || cmd[3] == L'\0') {
          CHAR16 *rest = cmd + 3;
          while (*rest == L' ') rest++;
          if (*rest == L'\0') { DoGopCommand(NULL); }
          else { DoGopCommand(rest); }
          continue;
        }
      }

      DoGopCommand(cmd);
    } // end interactive outer loop

    if (gPointerMode && Gop) StopPointerMode(Gop);
    Print(L"Exiting interactive mode.\n");
  }

  // cleanup image allocations before exit
  FreeImageList();
  FreeLoadedImage();

  return EFI_SUCCESS;
}



#if 0
// helloworld_gop_font_with_pointer.c
// Build with EDK2 (add to INF and compile).
#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Protocol/GraphicsOutput.h>
#include <Protocol/SimplePointer.h>
#include <Library/PrintLib.h>

// Place these includes near the top of your file if not already present:
//#include <Protocol/SimplePointer.h>
#include <Protocol/DevicePath.h>
#include <Protocol/DevicePathToText.h>

// small helpers for printing like the original code's TerminalPrintf:
#define TerminalPrintf(...) Print(__VA_ARGS__)

// Forward
EFI_STATUS GetGop(EFI_GRAPHICS_OUTPUT_PROTOCOL **OutGop);

// -------------------- existing globals & font code --------------------
STATIC BOOLEAN gFontBgTransparent = TRUE;
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL gFontBgColor = { .Blue = 0, .Green = 0, .Red = 0, .Reserved = 0 };
STATIC UINTN gFontScale = 2; // default scale factor
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL gFontFgColor = { .Blue = 0x00, .Green = 0xFF, .Red = 0xFF, .Reserved = 0 };
STATIC UINT8 gFontAlpha = 255; // 0..255

typedef struct { CHAR8 ch; UINT8 rows[7]; } GLYPH;

STATIC GLYPH sGlyphs[] = {
  { 'H', { 0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001 } },
  { 'E', { 0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111 } },
  { 'L', { 0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111 } },
  { 'O', { 0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110 } },
  { 'W', { 0b10001,0b10001,0b10001,0b10101,0b10101,0b11011,0b10001 } },
  { 'R', { 0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001 } },
  { 'D', { 0b11110,0b10001,0b10001,0b10001,0b10001,0b10001,0b11110 } },
  { '!', { 0b00100,0b00100,0b00100,0b00100,0b00100,0b00000,0b00100 } },
  { ',', { 0b00000,0b00000,0b00000,0b00000,0b00000,0b00100,0b01000 } },
  { ' ', { 0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000 } },
};

STATIC CONST GLYPH* FindGlyph(CHAR16 wch) {
  CHAR8 ch = (CHAR8)wch;
  if (ch >= 'a' && ch <= 'z') ch = (CHAR8)(ch - 'a' + 'A');
  for (UINTN i = 0; i < ARRAY_SIZE(sGlyphs); ++i) {
    if (sGlyphs[i].ch == ch) return &sGlyphs[i];
  }
  return NULL;
}

STATIC EFI_STATUS ReadPixel(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN X, UINTN Y, EFI_GRAPHICS_OUTPUT_BLT_PIXEL *Out) {
  if (!Gop || !Out) return EFI_INVALID_PARAMETER;
  return Gop->Blt(Gop, Out, EfiBltVideoToBltBuffer, X, Y, 0, 0, 1, 1, sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
}
STATIC EFI_STATUS WritePixel(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN X, UINTN Y, CONST EFI_GRAPHICS_OUTPUT_BLT_PIXEL *In) {
  if (!Gop || !In) return EFI_INVALID_PARAMETER;
  return Gop->Blt(Gop, (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)In, EfiBltBufferToVideo, 0, 0, X, Y, 1, 1, sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
}

STATIC VOID BlendPixel(const EFI_GRAPHICS_OUTPUT_BLT_PIXEL *src, EFI_GRAPHICS_OUTPUT_BLT_PIXEL *dst, UINT8 alpha) {
  UINTN inv = 255 - alpha;
  dst->Red   = (UINT8)((src->Red   * (UINT32)alpha + dst->Red   * inv) / 255);
  dst->Green = (UINT8)((src->Green * (UINT32)alpha + dst->Green * inv) / 255);
  dst->Blue  = (UINT8)((src->Blue  * (UINT32)alpha + dst->Blue  * inv) / 255);
}

// ---------- Buffered character drawing ----------
STATIC EFI_STATUS DrawCharBuffered(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN X, UINTN Y, CONST GLYPH *g) {
  if (!Gop || !g) return EFI_INVALID_PARAMETER;
  UINTN scale = gFontScale;
  UINTN w = 5 * scale;
  UINTN h = 7 * scale;
  UINTN PixelSize = sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
  UINTN RowBytes = w * PixelSize;
  UINTN BufSize = RowBytes * h;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL *Buf = AllocatePool(BufSize);
  if (!Buf) return EFI_OUT_OF_RESOURCES;

  EFI_STATUS Status = Gop->Blt(Gop, Buf, EfiBltVideoToBltBuffer, X, Y, 0, 0, w, h, RowBytes);
  if (EFI_ERROR(Status)) { FreePool(Buf); return Status; }

  for (UINTN row = 0; row < 7; ++row) {
    UINT8 bits = g->rows[row];
    for (UINTN col = 0; col < 5; ++col) {
      BOOLEAN bitSet = (bits >> (4 - col)) & 1;
      for (UINTN sy = 0; sy < scale; ++sy) {
        UINTN py = row * scale + sy;
        for (UINTN sx = 0; sx < scale; ++sx) {
          UINTN px = col * scale + sx;
          EFI_GRAPHICS_OUTPUT_BLT_PIXEL *pPixel = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)((UINT8*)Buf + (py * RowBytes) + (px * PixelSize));
          if (bitSet) {
            EFI_GRAPHICS_OUTPUT_BLT_PIXEL dst = *pPixel;
            EFI_GRAPHICS_OUTPUT_BLT_PIXEL src = gFontFgColor;
            EFI_GRAPHICS_OUTPUT_BLT_PIXEL out = dst;
            BlendPixel(&src, &out, gFontAlpha);
            *pPixel = out;
          } else {
            if (!gFontBgTransparent) {
              *pPixel = gFontBgColor;
            }
          }
        }
      }
    }
  }

  Status = Gop->Blt(Gop, Buf, EfiBltBufferToVideo, 0, 0, X, Y, w, h, RowBytes);
  FreePool(Buf);
  return Status;
}
STATIC EFI_STATUS DrawCharAt(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN x, UINTN y, CHAR16 ch) {
  const GLYPH *g = FindGlyph(ch);
  if (!g) return EFI_NOT_FOUND;
  return DrawCharBuffered(Gop, x, y, g);
}
STATIC EFI_STATUS DrawStringAt(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, UINTN x, UINTN y, CONST CHAR16 *Str) {
  if (!Gop || !Str) return EFI_INVALID_PARAMETER;
  UINTN cursorX = x;
  UINTN charWidth = 5 * gFontScale;
  UINTN spacing = gFontScale;
  while (*Str) {
    EFI_STATUS Status = DrawCharAt(Gop, cursorX, y, *Str);
    if (EFI_ERROR(Status)) return Status;
    cursorX += charWidth + spacing;
    ++Str;
  }
  return EFI_SUCCESS;
}

// Helper: print one simple pointer Mode structure
STATIC VOID PrintPointerModeInfo(EFI_SIMPLE_POINTER_PROTOCOL *Sp) {
  if (Sp == NULL) { Print(L"  <NULL SimplePointer protocol>\n"); return; }
  EFI_SIMPLE_POINTER_MODE *Mode = Sp->Mode;
  if (Mode == NULL) {
    Print(L"  SimplePointer Mode: <NULL>\n");
    return;
  }
  Print(L"  Mode: ResolutionX=%lu, ResolutionY=%lu, ResolutionZ=%lu, LeftButton=%u, RightButton=%u\n",
        (unsigned long)Mode->ResolutionX,
        (unsigned long)Mode->ResolutionY,
        (unsigned long)Mode->ResolutionZ,
        Mode->LeftButton ? 1 : 0,
        Mode->RightButton ? 1 : 0);
  Print(L"  WaitForInput event: %p\n", Sp->WaitForInput);
}


// Helpers
STATIC UINTN HexStrToUint(CONST CHAR16 *Str) {
  UINTN val = 0;
  while (*Str) {
    CHAR16 c = *Str++;
    if (c == L' ' || c == L'\0') break;
    val <<= 4;
    if (c >= L'0' && c <= L'9') val |= (UINTN)(c - L'0');
    else if (c >= L'A' && c <= L'F') val |= (UINTN)(10 + c - L'A');
    else if (c >= L'a' && c <= L'f') val |= (UINTN)(10 + c - L'a');
    else break;
  }
  return val;
}
STATIC UINTN StrToUintn(CONST CHAR16 *Str) {
  if (Str == NULL) return 0;
  UINTN Val = 0; BOOLEAN Any = FALSE;
  while (*Str) {
    if (*Str >= L'0' && *Str <= L'9') { Any = TRUE; Val = Val * 10 + (UINTN)(*Str - L'0'); Str++; } else break;
  }
  return Any ? Val : 0;
}

// -------------------- Pointer (mouse) support --------------------
// Globals to track pointer and cursor save buffer
STATIC EFI_SIMPLE_POINTER_PROTOCOL *gSimplePointer = NULL;
STATIC BOOLEAN gPointerMode = FALSE;
STATIC INTN gCursorX = -1, gCursorY = -1;
STATIC INTN gSavedCursorX = -1, gSavedCursorY = -1;
STATIC EFI_GRAPHICS_OUTPUT_BLT_PIXEL *gSavedCursorBg = NULL;
STATIC UINTN gCursorW = 12, gCursorH = 16; // cursor rectangle size

// a small arrow cursor (12x16) as bitmask rows (1 = pixel)
STATIC CONST UINT16 gCursorBitmap[16] = {
  0b100000000000,
  0b110000000000,
  0b111000000000,
  0b111100000000,
  0b111110000000,
  0b111111000000,
  0b111111100000,
  0b111111110000,
  0b111111100000,
  0b111101100000,
  0b111001100000,
  0b110001100000,
  0b100001100000,
  0b000001100000,
  0b000001000000,
  0b000000000000
};

// Helper: clamp
STATIC UINTN ClampU(UINTN v, UINTN lo, UINTN hi) { if (v < lo) return lo; if (v > hi) return hi; return v; }

// Restore previously saved background (if any)
STATIC VOID RestoreSavedCursor(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop) {
  if (!gSavedCursorBg || gSavedCursorX < 0 || gSavedCursorY < 0) return;
  UINTN PixelSize = sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
  UINTN RowBytes = gCursorW * PixelSize;
  // write saved background back
  (void)Gop->Blt(Gop, gSavedCursorBg, EfiBltBufferToVideo, 0, 0, (UINTN)gSavedCursorX, (UINTN)gSavedCursorY, gCursorW, gCursorH, RowBytes);
  FreePool(gSavedCursorBg);
  gSavedCursorBg = NULL;
  gSavedCursorX = gSavedCursorY = -1;
}

// Draw cursor at new position (saves the background where cursor will be drawn)
STATIC EFI_STATUS DrawCursorAt(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, INTN newX, INTN newY) {
  if (!Gop) return EFI_INVALID_PARAMETER;
  // clamp to screen
  UINTN screenW = Gop->Mode->Info->HorizontalResolution;
  UINTN screenH = Gop->Mode->Info->VerticalResolution;
  if (newX < 0) newX = 0;
  if (newY < 0) newY = 0;
  if ((UINTN)newX > screenW - gCursorW) newX = (INTN)(screenW - gCursorW);
  if ((UINTN)newY > screenH - gCursorH) newY = (INTN)(screenH - gCursorH);

  // If there is a saved background and position changed, restore it
  if (gSavedCursorBg && (gSavedCursorX != newX || gSavedCursorY != newY)) {
    RestoreSavedCursor(Gop);
  }

  // If no saved background for this position, save it
  if (!gSavedCursorBg) {
    UINTN PixelSize = sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL);
    UINTN RowBytes = gCursorW * PixelSize;
    UINTN BufSize = RowBytes * gCursorH;
    gSavedCursorBg = AllocatePool(BufSize);
    if (!gSavedCursorBg) return EFI_OUT_OF_RESOURCES;
    EFI_STATUS Status = Gop->Blt(Gop, gSavedCursorBg, EfiBltVideoToBltBuffer, (UINTN)newX, (UINTN)newY, 0, 0, gCursorW, gCursorH, RowBytes);
    if (EFI_ERROR(Status)) { FreePool(gSavedCursorBg); gSavedCursorBg = NULL; return Status; }
    gSavedCursorX = newX; gSavedCursorY = newY;

    // Prepare draw buffer by copying saved background and painting cursor pixels into it
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL *drawBuf = AllocatePool(BufSize);
    if (!drawBuf) { RestoreSavedCursor(Gop); return EFI_OUT_OF_RESOURCES; }
    CopyMem(drawBuf, gSavedCursorBg, BufSize);

    // Paint cursor pixels: use white color (or change to some color)
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL curColor = { .Blue = 0x00, .Green = 0x00, .Red = 0x00, .Reserved = 0 };
    // Use black arrow with white outline for visibility (draw twice with offsets)
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL outline = { .Blue = 0xFF, .Green = 0xFF, .Red = 0xFF, .Reserved = 0 };

    for (UINTN ry = 0; ry < gCursorH; ++ry) {
      UINT16 rowbits = (ry < 16) ? gCursorBitmap[ry] : 0;
      for (UINTN rx = 0; rx < gCursorW; ++rx) {
        BOOLEAN set = (rowbits & (1u << (gCursorW - 1 - rx))) != 0;
        if (set) {
          EFI_GRAPHICS_OUTPUT_BLT_PIXEL *p = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)((UINT8*)drawBuf + ry * RowBytes + rx * PixelSize);
          // simple overwrite with black; you could blend for alpha.
          *p = curColor;
        }
      }
    }
    // Optionally draw white outline pixels (simple neighbor pass)
    for (UINTN ry = 0; ry < gCursorH; ++ry) {
      UINT16 rowbits = (ry < 16) ? gCursorBitmap[ry] : 0;
      for (UINTN rx = 0; rx < gCursorW; ++rx) {
        BOOLEAN set = (rowbits & (1u << (gCursorW - 1 - rx))) != 0;
        if (set) {
          // neighbors
          for (INTN ny = -1; ny <= 1; ++ny) for (INTN nx = -1; nx <= 1; ++nx) {
            INTN px = (INTN)rx + nx;
            INTN py = (INTN)ry + ny;
            if (px >= 0 && py >= 0 && (UINTN)px < gCursorW && (UINTN)py < gCursorH) {
              UINT16 nrow = (py < 16) ? gCursorBitmap[py] : 0;
              BOOLEAN nset = (nrow & (1u << (gCursorW - 1 - px))) != 0;
              if (!nset) {
                EFI_GRAPHICS_OUTPUT_BLT_PIXEL *p = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL*)((UINT8*)drawBuf + py * RowBytes + px * sizeof(EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
                *p = outline;
              }
            }
          }
        }
      }
    }

    // Write composed cursor image to video
    EFI_STATUS s = Gop->Blt(Gop, drawBuf, EfiBltBufferToVideo, 0, 0, (UINTN)newX, (UINTN)newY, gCursorW, gCursorH, RowBytes);
    FreePool(drawBuf);
    if (EFI_ERROR(s)) { RestoreSavedCursor(Gop); return s; }
  }

  gCursorX = newX; gCursorY = newY;
  return EFI_SUCCESS;
}

// Start pointer mode: locate protocol and draw initial cursor
STATIC VOID StartPointerMode() {
  if (gPointerMode) return;
  EFI_STATUS Status = gBS->LocateProtocol(&gEfiSimplePointerProtocolGuid, NULL, (VOID**)&gSimplePointer);
  if (EFI_ERROR(Status) || gSimplePointer == NULL) {
    TerminalPrintf(L"pointer_mode: SimplePointer protocol not found: %r\n", Status);
    gSimplePointer = NULL;
    gPointerMode = FALSE;
    return;
  }
  gPointerMode = TRUE;
  // initial position will be set later on first pointer event; but we can set center
  // We'll let interactive loop call DrawCursorAt on first pointer event.
  TerminalPrintf(L"pointer_mode: enabled (SimplePointer detected)\n");
}

// Stop pointer mode and restore any saved cursor area
STATIC VOID StopPointerMode(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop) {
  if (!gPointerMode) return;
  // restore saved cursor background if any
  RestoreSavedCursor(Gop);
  gPointerMode = FALSE;
  gSimplePointer = NULL;
  TerminalPrintf(L"pointer_mode: disabled\n");
}

// Draw the X/Y log on screen (top-left). This uses a small background rectangle and DrawStringAt.
STATIC VOID DrawPointerLog(EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop, INTN x, INTN y) {
  if (!Gop) return;
  // prepare formatted text "X:xxxx Y:yyyy"
  CHAR16 buf[64];
  UnicodeSPrint(buf, sizeof(buf), L"X:%d Y:%d", x, y);
  // Draw opaque background rectangle for readability
  UINTN textW = (5 * gFontScale) * StrLen(buf) + gFontScale * StrLen(buf); // rough estimate (5px per char + spacing)
  UINTN textH = 7 * gFontScale;
  // clamp area
  UINTN sx = 8, sy = 8;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL bg = { .Blue = 0x33, .Green = 0x33, .Red = 0x33, .Reserved = 0 };
  Gop->Blt(Gop, &bg, EfiBltVideoFill, 0, 0, sx-2, sy-2, textW+4, textH+4, 0);
  // Temporarily force opaque background color for DrawString
  BOOLEAN oldTransparent = gFontBgTransparent;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL oldBg = gFontBgColor;
  gFontBgTransparent = FALSE;
  gFontBgColor.Red = 0x33; gFontBgColor.Green = 0x33; gFontBgColor.Blue = 0x33;
  DrawStringAt(Gop, sx, sy, buf);
  // restore
  gFontBgTransparent = oldTransparent;
  gFontBgColor = oldBg;
}

// -------------------- DoGopCommand updated with pointer_mode --------------------
STATIC VOID DoGopCommand(CONST CHAR16 *Arg) {
  if (Arg == NULL) {
    TerminalPrintf(L"gop: missing argument (list|set|bg|fontbg|fontfg|fontsize|draw|blendtest|pointer_mode)\n");
    return;
  }

  CHAR16 Work[512];
  StrCpyS(Work, ARRAY_SIZE(Work), Arg);
  CHAR16 *Sub = Work;
  CHAR16 *Rest = StrStr(Sub, L" ");
  if (Rest) { *Rest = L'\0'; Rest++; while (*Rest == L' ') Rest++; }

	/*
  if (StrCmp(Sub, L"pointer_mode") == 0) {
    // pointer_mode on|off
    if (Rest == NULL) { TerminalPrintf(L"gop pointer_mode: missing arg (on|off)\n"); return; }
    if (StrCmp(Rest, L"on") == 0) {
      StartPointerMode();
      return;
    } else if (StrCmp(Rest, L"off") == 0) {
      EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
      if (!EFI_ERROR(GetGop(&Gop))) StopPointerMode(Gop);
      else StopPointerMode(NULL);
      return;
    } else {
      TerminalPrintf(L"gop pointer_mode: unknown arg '%s'\n", Rest);
      return;
    }
  }
  */
  
  // ---------- replace pointer_mode handling with this block ----------
  if (StrCmp(Sub, L"pointer_mode") == 0) {
    // pointer_mode on|off|device
    if (Rest == NULL) { TerminalPrintf(L"gop pointer_mode: missing arg (on|off|device)\n"); return; }

    if (StrCmp(Rest, L"on") == 0) {
      // existing logic (preserve)
      StartPointerMode();
      return;
    } else if (StrCmp(Rest, L"off") == 0) {
      EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
      if (!EFI_ERROR(GetGop(&Gop))) StopPointerMode(Gop);
      else StopPointerMode(NULL);
      return;
    } else if (StrCmp(Rest, L"device") == 0) {
      // New: enumerate SIMPLE POINTER handles and print info + monitor first device.
      EFI_STATUS Status;
      EFI_HANDLE *Handles = NULL;
      UINTN HandleCount = 0;

      Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiSimplePointerProtocolGuid, NULL, &HandleCount, &Handles);
      if (EFI_ERROR(Status) || HandleCount == 0) {
        TerminalPrintf(L"gop pointer_mode device: no SimplePointer handles found (%r)\n", Status);
        if (Handles) { FreePool(Handles); }
        return;
      }

      TerminalPrintf(L"gop pointer_mode device: found %u SimplePointer handle(s)\n", HandleCount);

      // Try to locate DevicePathToText protocol once for converting device paths (optional)
      EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpText = NULL;
      Status = gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID**)&DpText);
      if (EFI_ERROR(Status)) DpText = NULL;

      // Print info for each handle
      for (UINTN i = 0; i < HandleCount; ++i) {
        EFI_HANDLE h = Handles[i];
        TerminalPrintf(L"Handle[%u] = %p\n", i, h);

        // Try to get SimplePointer protocol on this handle
        EFI_SIMPLE_POINTER_PROTOCOL *Sp = NULL;
        Status = gBS->HandleProtocol(h, &gEfiSimplePointerProtocolGuid, (VOID**)&Sp);
        if (EFI_ERROR(Status) || Sp == NULL) {
          TerminalPrintf(L"  Could not get SimplePointer protocol: %r\n", Status);
        } else {
          PrintPointerModeInfo(Sp);
        }

        // Try to get a device path for this handle (optional, might fail)
        EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
        Status = gBS->HandleProtocol(h, &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
        if (!EFI_ERROR(Status) && DevPath != NULL && DpText != NULL) {
          CHAR16 *DpStr = DpText->ConvertDevicePathToText(DevPath, TRUE, TRUE);
          if (DpStr) {
            TerminalPrintf(L"  DevicePath: %s\n", DpStr);
            FreePool(DpStr);
          } else {
            TerminalPrintf(L"  DevicePath: (ConvertDevicePathToText returned NULL)\n");
          }
        } else {
          TerminalPrintf(L"  DevicePath: not available (HandleProtocol returned %r or converter not found)\n", Status);
        }
      } // end for handles

      // Monitor the first SimplePointer device (if present) and print live movement until a key is pressed
      {
        EFI_SIMPLE_POINTER_PROTOCOL *MonitorSp = NULL;
        EFI_HANDLE monitorHandle = Handles[0];
        Status = gBS->HandleProtocol(monitorHandle, &gEfiSimplePointerProtocolGuid, (VOID**)&MonitorSp);
        if (EFI_ERROR(Status) || MonitorSp == NULL) {
          TerminalPrintf(L"gop pointer_mode device: cannot open first SimplePointer for monitoring: %r\n", Status);
          FreePool(Handles);
          return;
        }

        TerminalPrintf(L"\nMonitoring SimplePointer on handle %p. Move the device to see RelativeMovement values.\n", monitorHandle);
        TerminalPrintf(L"Press any key to stop monitoring.\n");

        // maintain cumulative position (relative -> simulate absolute)
        //INT64 posX = (INT64)(Gop->Mode->Info->HorizontalResolution / 2);
        //INT64 posY = (INT64)(Gop->Mode->Info->VerticalResolution / 2);
		
		INT64 posY;
		INT64 posX;
        
		EFI_GRAPHICS_OUTPUT_PROTOCOL *GopLocal = NULL;
		if (EFI_ERROR(GetGop(&GopLocal)) || GopLocal == NULL) {
		  // GOP not available — start at (0,0) to avoid using an undeclared variable
		  posX = (INT64)0;
		  posY = (INT64)0;
		} else {
		  // center start position on the screen
		  posX = (INT64)(GopLocal->Mode->Info->HorizontalResolution) / 2;
		  posY = (INT64)(GopLocal->Mode->Info->VerticalResolution) / 2;
		}

		
		EFI_SIMPLE_POINTER_STATE State;
        EFI_STATUS getst;

        // Use WaitForInput if available to wait for pointer changes; also check keyboard to break.
        BOOLEAN useWaitForInput = (MonitorSp->WaitForInput != NULL);
        while (TRUE) {
          // If WaitForInput is present, wait on both events: key and pointer event.
          if (useWaitForInput) {
            EFI_EVENT ev[2];
            ev[0] = gST->ConIn->WaitForKey;
            ev[1] = MonitorSp->WaitForInput;
            UINTN idx;
            Status = gBS->WaitForEvent(2, ev, &idx);
            if (EFI_ERROR(Status)) break;
            if (idx == 0) {
              // key pressed -> drain key and break
              EFI_INPUT_KEY K;
              while (gST->ConIn->ReadKeyStroke(gST->ConIn, &K) == EFI_SUCCESS) { /* consume */ }
              break;
            }
            // idx == 1 -> pointer signaled, read state
            getst = MonitorSp->GetState(MonitorSp, &State);
            if (!EFI_ERROR(getst)) {
              // Update cumulative and print
              posX += (INT64)State.RelativeMovementX;
              posY += (INT64)State.RelativeMovementY;
              TerminalPrintf(L"Pointer event: RelX=%d RelY=%d RelZ=%d Left=%u Right=%u  PosX=%ld PosY=%ld\n",
                             (INT32)State.RelativeMovementX, (INT32)State.RelativeMovementY, (INT32)State.RelativeMovementZ,
                             State.LeftButton ? 1 : 0, State.RightButton ? 1 : 0,
                             (long)posX, (long)posY);
            } else if (getst == EFI_NOT_READY) {
              // no change; continue
            } else {
              TerminalPrintf(L"  GetState returned %r\n", getst);
            }
            // loop
            continue;
          } // end useWaitForInput

          // Otherwise, fallback: polling loop (check keyboard to exit)
          // check key
          EFI_INPUT_KEY K;
          if (gST->ConIn->ReadKeyStroke(gST->ConIn, &K) == EFI_SUCCESS) {
            // consume any additional keys
            while (gST->ConIn->ReadKeyStroke(gST->ConIn, &K) == EFI_SUCCESS) { }
            break;
          }
          // attempt GetState
          getst = MonitorSp->GetState(MonitorSp, &State);
          if (!EFI_ERROR(getst)) {
            posX += (INT64)State.RelativeMovementX;
            posY += (INT64)State.RelativeMovementY;
            TerminalPrintf(L"[poll] Pointer: RelX=%d RelY=%d Left=%u Right=%u  PosX=%ld PosY=%ld\n",
                           (INT32)State.RelativeMovementX, (INT32)State.RelativeMovementY,
                           State.LeftButton ? 1 : 0, State.RightButton ? 1 : 0,
                           (long)posX, (long)posY);
          } else if (getst != EFI_NOT_READY) {
            TerminalPrintf(L"[poll] GetState returned %r\n", getst);
          }
          // small sleep to avoid busy loop (100ms)
          gBS->Stall(100000);
        } // end monitor loop

        TerminalPrintf(L"Monitoring ended (key pressed or error). Returning to prompt.\n");
      }

      FreePool(Handles);
      return;
    } else {
      TerminalPrintf(L"gop pointer_mode: unknown arg '%s' (use on|off|device)\n", Rest);
      return;
    }
  }

  // ... rest of your DoGopCommand implementation (list,set,bg,fontbg,fontfg,fontsize,draw,blendtest)
  // For brevity we re-use the previous implementation for those commands.
  // (Insert the full previous DoGopCommand body here — but since we already have it below in the file
  //  we'll avoid duplication. For this snippet, assume the rest of commands are present.)
  // To keep the answer concise here, I'll dispatch back to the rest of the commands by duplicating
  // the previous parsing logic (starting from list). — continue with the other commands:

  if (StrCmp(Sub, L"list") == 0) {
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop list: GOP not available: %r\n", Status); return; }
    TerminalPrintf(L"Mode Count: %u, CurrentMode: %u\n", Gop->Mode->MaxMode, Gop->Mode->Mode);
    for (UINT32 i = 0; i < Gop->Mode->MaxMode; ++i) {
      EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *Info;
      UINTN SizeOfInfo;
      Status = Gop->QueryMode(Gop, i, &SizeOfInfo, &Info);
      if (EFI_ERROR(Status)) continue;
      TerminalPrintf(L"  [%u] %ux%u, PixelFormat %u, PixelsPerScanLine %u\n",
                     i, Info->HorizontalResolution, Info->VerticalResolution, Info->PixelFormat, Info->PixelsPerScanLine);
    }
    return;
  } else if (StrCmp(Sub, L"set") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop set: missing mode index\n"); return; }
    UINTN idx = StrToUintn(Rest);
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop set: GOP not available: %r\n", Status); return; }
    if (idx >= Gop->Mode->MaxMode) { TerminalPrintf(L"gop set: invalid mode index %u\n", idx); return; }
    Status = Gop->SetMode(Gop, (UINT32)idx);
    if (EFI_ERROR(Status)) TerminalPrintf(L"gop set: SetMode failed: %r\n", Status); else TerminalPrintf(L"gop set: mode changed to %u\n", idx);
    return;
  } else if (StrCmp(Sub, L"bg") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop bg: missing hex color (RRGGBB)\n"); return; }
    UINTN val = HexStrToUint(Rest);
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop bg: GOP not available: %r\n", Status); return; }
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL Color;
    Color.Red = (UINT8)((val >> 16) & 0xFF);
    Color.Green = (UINT8)((val >> 8) & 0xFF);
    Color.Blue = (UINT8)(val & 0xFF);
    Color.Reserved = 0;
    Status = Gop->Blt(Gop, &Color, EfiBltVideoFill, 0, 0, 0, 0, Gop->Mode->Info->HorizontalResolution, Gop->Mode->Info->VerticalResolution, 0);
    if (EFI_ERROR(Status)) TerminalPrintf(L"gop bg: BLT failed: %r\n", Status); else TerminalPrintf(L"gop bg: background set to #%06x\n", val);
    return;
  } else if (StrCmp(Sub, L"fontbg") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop fontbg: missing argument (transparent|opaque|color RRGGBB)\n"); return; }
    CHAR16 *arg1 = Rest;
    CHAR16 *arg2 = StrStr(arg1, L" ");
    if (arg2) { *arg2 = L'\0'; arg2++; while (*arg2 == L' ') arg2++; }
    if (StrCmp(arg1, L"transparent") == 0) {
      gFontBgTransparent = TRUE; TerminalPrintf(L"gop fontbg: transparent enabled\n"); return;
    } else if (StrCmp(arg1, L"opaque") == 0) {
      gFontBgTransparent = FALSE; TerminalPrintf(L"gop fontbg: transparent disabled (opaque)\n"); return;
    } else if (StrCmp(arg1, L"color") == 0) {
      if (arg2 == NULL) { TerminalPrintf(L"gop fontbg: missing hex color after 'color'\n"); return; }
      UINTN val = HexStrToUint(arg2);
      gFontBgColor.Red = (UINT8)((val >> 16) & 0xFF);
      gFontBgColor.Green = (UINT8)((val >> 8) & 0xFF);
      gFontBgColor.Blue = (UINT8)(val & 0xFF);
      gFontBgColor.Reserved = 0;
      TerminalPrintf(L"gop fontbg: background color set to #%06x\n", val);
      return;
    } else {
      TerminalPrintf(L"gop fontbg: unknown option '%s'\n", arg1); return;
    }
  } else if (StrCmp(Sub, L"fontfg") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop fontfg: missing argument (color RRGGBB|alpha N)\n"); return; }
    CHAR16 *arg1 = Rest;
    CHAR16 *arg2 = StrStr(arg1, L" ");
    if (arg2) { *arg2 = L'\0'; arg2++; while (*arg2 == L' ') arg2++; }
    if (StrCmp(arg1, L"color") == 0) {
      if (arg2 == NULL) { TerminalPrintf(L"gop fontfg: missing hex color after 'color'\n"); return; }
      UINTN val = HexStrToUint(arg2);
      gFontFgColor.Red = (UINT8)((val >> 16) & 0xFF);
      gFontFgColor.Green = (UINT8)((val >> 8) & 0xFF);
      gFontFgColor.Blue = (UINT8)(val & 0xFF);
      gFontFgColor.Reserved = 0;
      TerminalPrintf(L"gop fontfg: foreground color set to #%06x\n", val);
      return;
    } else if (StrCmp(arg1, L"alpha") == 0) {
      if (arg2 == NULL) { TerminalPrintf(L"gop fontfg: missing alpha value after 'alpha'\n"); return; }
      UINTN a = StrToUintn(arg2);
      if (a > 255) a = 255;
      gFontAlpha = (UINT8)a;
      TerminalPrintf(L"gop fontfg: alpha set to %u\n", gFontAlpha);
      return;
    } else {
      TerminalPrintf(L"gop fontfg: unknown option '%s'\n", arg1); return;
    }
  } else if (StrCmp(Sub, L"fontsize") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop fontsize: missing scale factor\n"); return; }
    UINTN s = StrToUintn(Rest);
    if (s < 1) s = 1;
    gFontScale = s; TerminalPrintf(L"gop fontsize: scale set to %u\n", gFontScale); return;
  } else if (StrCmp(Sub, L"draw") == 0) {
    if (Rest == NULL) { TerminalPrintf(L"gop draw: missing args\n"); return; }
    CHAR16 *p = Rest;
    CHAR16 *p2 = StrStr(p, L" ");
    if (!p2) { TerminalPrintf(L"gop draw: missing Y and text\n"); return; }
    *p2 = L'\0'; p2++; while (*p2 == L' ') p2++;
    UINTN x = StrToUintn(p);
    CHAR16 *p3 = StrStr(p2, L" ");
    if (!p3) { TerminalPrintf(L"gop draw: missing text\n"); return; }
    *p3 = L'\0'; p3++; while (*p3 == L' ') p3++;
    UINTN y = StrToUintn(p2);
    CHAR16 *txt = p3;
    if (*txt == L'"') {
      txt++;
      CHAR16 *q = StrStr(txt, L"\"");
      if (q) *q = L'\0';
    }
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop draw: GOP not available: %r\n", Status); return; }
    Status = DrawStringAt(Gop, x, y, txt);
    if (EFI_ERROR(Status)) TerminalPrintf(L"gop draw: draw failed: %r\n", Status); else TerminalPrintf(L"gop draw: drew '%s' at %u,%u\n", txt, x, y);
    return;
  } else if (StrCmp(Sub, L"blendtest") == 0) {
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_STATUS Status = GetGop(&Gop);
    if (EFI_ERROR(Status)) { TerminalPrintf(L"gop blendtest: GOP not available: %r\n", Status); return; }
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL blue = { .Blue = 0xFF, .Green = 0x00, .Red = 0x00, .Reserved = 0 };
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL red  = { .Blue = 0x00, .Green = 0x00, .Red = 0xFF, .Reserved = 0 };
    UINTN x0 = 50, y0 = 50;
    for (UINTN y = y0; y < y0 + 80; ++y) for (UINTN x = x0; x < x0 + 160; ++x) WritePixel(Gop, x, y, &blue);
    for (UINTN y = y0 + 20; y < y0 + 100; ++y) for (UINTN x = x0 + 40; x < x0 + 200; ++x) {
      EFI_GRAPHICS_OUTPUT_BLT_PIXEL dst;
      if (EFI_ERROR(ReadPixel(Gop, x, y, &dst))) continue;
      EFI_GRAPHICS_OUTPUT_BLT_PIXEL out = dst;
      BlendPixel(&red, &out, 128);
      WritePixel(Gop, x, y, &out);
    }
    TerminalPrintf(L"gop blendtest: finished\n");
    return;
  }

  TerminalPrintf(L"gop: unknown subcommand '%s'\n", Sub);
}

// -------------------- GetGop --------------------
EFI_STATUS GetGop(EFI_GRAPHICS_OUTPUT_PROTOCOL **OutGop) {
  return gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID**)OutGop);
}

// -------------------- Entry point & interactive loop (with pointer support) --------------------
EFI_STATUS EFIAPI UefiMain(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable) {
  EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
  EFI_STATUS Status = GetGop(&Gop);
  if (EFI_ERROR(Status)) {
    Print(L"helloworld: GOP not available: %r\n", Status);
  } else {
    Print(L"helloworld: GOP mode: %ux%u (Mode %u)\n",
          Gop->Mode->Info->HorizontalResolution, Gop->Mode->Info->VerticalResolution, Gop->Mode->Mode);
  }

  Print(L"Simple interactive demo (examples):\n");
  Print(L"  gop list\n");
  Print(L"  gop set 1\n");
  Print(L"  gop bg FF00FF\n");
  Print(L"  gop fontbg transparent\n");
  Print(L"  gop fontsize 3\n");
  Print(L"  gop fontfg color FFFF00\n");
  Print(L"  gop draw 50 50 \"HELLO, WORLD!\"\n");
  Print(L"  gop blendtest\n");
  Print(L"  gop pointer_mode on\n\n");

  // Demo initial draw
  gFontBgTransparent = TRUE;
  gFontScale = 3;
  gFontFgColor.Red = 0xFF; gFontFgColor.Green = 0xFF; gFontFgColor.Blue = 0x00;
  gFontAlpha = 255;
  if (!EFI_ERROR(GetGop(&Gop))) {
    EFI_GRAPHICS_OUTPUT_BLT_PIXEL bg = { .Blue = 0x33, .Green = 0x33, .Red = 0x33, .Reserved = 0 };
    Gop->Blt(Gop, &bg, EfiBltVideoFill, 0,0,0,0,Gop->Mode->Info->HorizontalResolution,Gop->Mode->Info->VerticalResolution,0);
    DrawStringAt(Gop, 60, 40, L"HELLO, WORLD!");
  }

  // Example commands
  DoGopCommand(L"list");
  DoGopCommand(L"fontsize 2");
  DoGopCommand(L"fontbg color FF0000");
  DoGopCommand(L"fontbg opaque");
  DoGopCommand(L"draw 60 120 \"HELLO, WORLD!\"");

  // Interactive command loop with pointer event integration
  {
    BOOLEAN Done = FALSE;
    CHAR16 Line[512];
    UINTN Len;

    Print(L"\nEntering interactive mode. Type 'exit' to quit.\n");
    while (!Done) {
      // Build event list: keyboard plus optionally pointer
      EFI_EVENT Events[2];
      UINTN EventCount = 0;
      Events[EventCount++] = gST->ConIn->WaitForKey;
      if (gPointerMode && gSimplePointer && gSimplePointer->WaitForInput) {
        Events[EventCount++] = gSimplePointer->WaitForInput;
      }

      // Prompt & reset line buffer
      Print(L"> ");
      Len = 0;
      Line[0] = L'\0';

      // We'll loop, waiting for either keyboard or pointer events.
      while (TRUE) {
        UINTN Index;
        EFI_STATUS WaitStatus = gBS->WaitForEvent(EventCount, Events, &Index);
        if (EFI_ERROR(WaitStatus)) break;

        // If pointer event triggered (Index == 1 when present), process pointer movement first
        if (gPointerMode && EventCount == 2 && Index == 1) {
          // pointer event
          EFI_SIMPLE_POINTER_STATE State;
          if (gSimplePointer && gSimplePointer->GetState && !EFI_ERROR(gSimplePointer->GetState(gSimplePointer, &State))) {
            // Update cursor position (relative)
            // SimplePointer uses RelativeMovementX/Y (INT32) in many implementations.
            INTN dx = (INTN)State.RelativeMovementX;
            INTN dy = (INTN)State.RelativeMovementY;
            if (dx != 0 || dy != 0 || gCursorX < 0) {
              INTN newX = (gCursorX < 0) ? (INTN)(Gop->Mode->Info->HorizontalResolution / 2) : gCursorX + dx;
              INTN newY = (gCursorY < 0) ? (INTN)(Gop->Mode->Info->VerticalResolution / 2) : gCursorY + dy;
              // clamp
              if ((UINTN)newX >= Gop->Mode->Info->HorizontalResolution) newX = (INTN)(Gop->Mode->Info->HorizontalResolution - 1);
              if ((UINTN)newY >= Gop->Mode->Info->VerticalResolution) newY = (INTN)(Gop->Mode->Info->VerticalResolution - 1);
              DrawCursorAt(Gop, newX, newY);
              DrawPointerLog(Gop, newX, newY);
            }
          }
          // continue waiting for keyboard or more pointer events
          continue;
        }

        // Otherwise keyboard event (Index == 0)
        EFI_INPUT_KEY Key;
        if (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_SUCCESS) {
          // Handle Enter
          if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN || Key.UnicodeChar == CHAR_LINEFEED) {
            Print(L"\r\n");
            break;
          }
          // Backspace
          if (Key.UnicodeChar == CHAR_BACKSPACE) {
            if (Len > 0) {
              Len--;
              Line[Len] = L'\0';
              Print(L"\b \b");
            }
            continue;
          }
          if (Key.UnicodeChar == 0) {
            continue;
          }
          if (Len + 1 < ARRAY_SIZE(Line)) {
            Line[Len++] = Key.UnicodeChar;
            Line[Len] = L'\0';
            Print(L"%c", Key.UnicodeChar);
          }
        }
      } // end inner wait loop (we have a full line)

      // Trim leading spaces
      CHAR16 *cmd = Line;
      while (*cmd == L' ') cmd++;

      if (StrLen(cmd) == 0) continue;

      if (StrCmp(cmd, L"exit") == 0) {
        Done = TRUE;
        continue;
      }

      // If the input starts with "gop", strip it
      if (StrnCmp(cmd, L"gop", 3) == 0) {
        if (cmd[3] == L' ' || cmd[3] == L'\0') {
          CHAR16 *rest = cmd + 3;
          while (*rest == L' ') rest++;
          if (*rest == L'\0') { DoGopCommand(NULL); }
          else { DoGopCommand(rest); }
          continue;
        }
      }

      // Otherwise pass-line to command parser
      DoGopCommand(cmd);
    } // end interactive outer loop

    // On exit, restore cursor background if active
    if (gPointerMode && Gop) StopPointerMode(Gop);
    Print(L"Exiting interactive mode.\n");
  }

  return EFI_SUCCESS;
}
#endif
