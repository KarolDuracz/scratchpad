/** HelloWorld.c
  Advanced HelloWorld terminal with:
    - loadshell/loadimg/listapps
    - bgtext (background under text)
    - text buffer with PageUp/PageDown scrolling
    - setbuf to control buffer size
*/

/*
#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DevicePathLib.h>
#include <Protocol/LoadedImage.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/GraphicsOutput.h>
#include <Protocol/SimpleTextIn.h>    // SCAN_PAGE_UP / SCAN_PAGE_DOWN
#include <Guid/FileInfo.h>
#include <Library/UefiApplicationEntryPoint.h>
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DevicePathLib.h>
/* Added headers so UnicodeVSPrint/Ascii helpers and va_list exist */
#include <Library/PrintLib.h>    /* UnicodeVSPrint */
//#include <Library/AsciiLib.h>    /* UnicodeStrToAsciiStrS */
#include <stdarg.h>             /* VA_LIST, VA_START, VA_END */
#include <Protocol/LoadedImage.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/GraphicsOutput.h>
#include <Protocol/SimpleTextIn.h>    // SCAN_PAGE_UP / SCAN_PAGE_DOWN
#include <Guid/FileInfo.h>
#include <Library/UefiApplicationEntryPoint.h>

#define LINE_BUFFER_SIZE 1024
#define FILE_READ_BUFSIZE 0x1000

// text buffer defaults
#define TEXT_BUFFER_DEFAULT_LINES 400
#define TEXT_LINE_MAX_CHARS 1024
#define CHAR_HEIGHT_ESTIMATE 16 // pixels; approximate for "bgtext" height calculation

STATIC EFI_FILE_PROTOCOL *gRoot = NULL;
STATIC EFI_FILE_PROTOCOL *gCurDir = NULL;
STATIC CHAR16 *gCurPath = NULL; // current path, always begins with '\'
STATIC EFI_HANDLE mImageHandle = NULL;
STATIC EFI_HANDLE mDeviceHandle = NULL;

/* Text buffer */
STATIC CHAR16 **TextBuffer = NULL;
STATIC UINTN TextBufferLines = 0;
STATIC UINTN TextBufferHead = 0;   // index of oldest line in buffer
STATIC UINTN TextBufferCount = 0;  // number of stored lines (<= TextBufferLines)
STATIC INTN  ViewOffset = 0;       // 0 = show newest (live); >0 means scrolled up by that many lines

//
// Forward declarations
//
STATIC CHAR16 * DupStr(CONST CHAR16 *S);
STATIC VOID FreeStr(CHAR16 *S);
STATIC EFI_STATUS OpenPathAsFile(EFI_FILE_PROTOCOL **StartDir, CONST CHAR16 *Path, EFI_FILE_PROTOCOL **OutFile, BOOLEAN MustBeDir);
STATIC VOID PrintPrompt(VOID);
STATIC VOID ReadLine(CHAR16 *Buffer, UINTN BufferSize);
STATIC VOID DoCmdDir(CONST CHAR16 *Arg);
STATIC VOID DoCmdPwd(VOID);
STATIC VOID DoCmdCd(CONST CHAR16 *Arg);
STATIC VOID DoCmdLoadShell(CONST CHAR16 *Arg);
STATIC VOID DoCmdLoadImg(CONST CHAR16 *Arg);
STATIC VOID DoCmdListApps(VOID);
STATIC VOID PrintHelp(VOID);
STATIC VOID CloseCurrentDir(VOID);
STATIC CHAR16 *BuildAbsolutePath(CONST CHAR16 *Arg);
STATIC VOID DoGopCommand(CONST CHAR16 *Arg);
STATIC EFI_STATUS GetGop(EFI_GRAPHICS_OUTPUT_PROTOCOL **Out);

/* Buffer helpers */
STATIC EFI_STATUS InitTextBuffer(UINTN Lines);
STATIC VOID FreeTextBuffer(VOID);
STATIC VOID TerminalAddLine(CONST CHAR16 *Line);
//STATIC VOID TerminalPrintf(CONST CHAR16 *Fmt, ...) PRINTF_ATTR(1, 2);
/* forward declaration for TerminalPrintf (no PRINTF_ATTR here) */
STATIC VOID TerminalPrintf(CONST CHAR16 *Fmt, ...);

STATIC VOID RefreshScreenWithBuffer(VOID);
STATIC VOID ScrollBufferUp(UINTN lines);
STATIC VOID ScrollBufferDown(UINTN lines);
STATIC VOID SetTextBufferSize(UINTN lines);

/* utility */
STATIC UINTN StrToUintn(CONST CHAR16 *Str);
STATIC VOID DoBgText(CONST CHAR16 *Arg);

//
// HexStrToUint: parse up to 8 hex digits from a CHAR16 string (optional "0x" or "#" prefix).
// Stops at first non-hex character. Returns 0 if none found.
STATIC UINTN
HexStrToUint(CONST CHAR16 *Str)
{
  if (Str == NULL) {
    return 0;
  }

  // skip leading spaces
  while (*Str == L' ' || *Str == L'\t') Str++;

  // optional "0x" or "0X" or '#'
  if (Str[0] == L'0' && (Str[1] == L'x' || Str[1] == L'X')) Str += 2;
  else if (Str[0] == L'#') Str++;

  UINTN Val = 0;
  BOOLEAN Any = FALSE;

  while (*Str != L'\0') {
    CHAR16 c = *Str;
    UINTN d;
    if (c >= L'0' && c <= L'9') d = (UINTN)(c - L'0');
    else if (c >= L'a' && c <= L'f') d = (UINTN)(10 + (c - L'a'));
    else if (c >= L'A' && c <= L'F') d = (UINTN)(10 + (c - L'A'));
    else break;

    Any = TRUE;
    // avoid overflow - allow up to 32 bits, but shifting will naturally wrap on huge numbers (caller expects 24-bit colors)
    Val = (Val << 4) | d;
    Str++;
  }

  return Any ? Val : 0;
}



EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS                    Status;
  EFI_LOADED_IMAGE_PROTOCOL     *ThisLoadedImage;
  EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *SimpleFs = NULL;
  CHAR16 Line[LINE_BUFFER_SIZE];
  CHAR16 *Cmd, *Arg;

  mImageHandle = ImageHandle;

  // Initialize text buffer
  Status = InitTextBuffer(TEXT_BUFFER_DEFAULT_LINES);
  if (EFI_ERROR(Status)) {
    // Fatal - cannot initialize buffer, but still try to run minimal output
    Print(L"HelloWorld: cannot init text buffer: %r\n", Status);
    return Status;
  }

  TerminalPrintf(L"HelloWorld: advanced terminal starting...\n");

  // get loaded image protocol
  Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&ThisLoadedImage);
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"HelloWorld: ERROR - no LoadedImageProtocol: %r\n", Status);
    FreeTextBuffer();
    return Status;
  }
  // remember device handle (volume device handle) to build device paths later
  mDeviceHandle = ThisLoadedImage->DeviceHandle;

  // get simple file system protocol on the same device the image was loaded from
  Status = gBS->HandleProtocol(ThisLoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&SimpleFs);
  if (EFI_ERROR(Status) || SimpleFs == NULL) {
    TerminalPrintf(L"HelloWorld: ERROR - no SimpleFileSystem on device: %r\n", Status);
    FreeTextBuffer();
    return Status;
  }

  // open the root volume
  Status = SimpleFs->OpenVolume(SimpleFs, &gRoot);
  if (EFI_ERROR(Status) || gRoot == NULL) {
    TerminalPrintf(L"HelloWorld: ERROR - OpenVolume failed: %r\n", Status);
    FreeTextBuffer();
    return Status;
  }

  // set current dir = root, current path = "\"
  gCurDir = gRoot;
  gCurPath = DupStr(L"\\");

  TerminalPrintf(L"HelloWorld: filesystem ready. Type 'help' for commands.\n");

  // Main command loop
  for (;;) {
    PrintPrompt();
    ReadLine(Line, ARRAY_SIZE(Line));

    // strip leading spaces
    Cmd = Line;
    while (*Cmd == L' ' || *Cmd == L'\t') Cmd++;

    // empty?
    if (*Cmd == L'\0') {
      continue;
    }

    // split into command and arg
    Arg = StrStr(Cmd, L" ");
    if (Arg != NULL) {
      *Arg = L'\0';
      Arg++;
      while (*Arg == L' ' || *Arg == L'\t') Arg++;
      if (*Arg == L'\0') Arg = NULL;
    }

    // tolower the command first (simple)
    for (CHAR16 *p = Cmd; *p; ++p) {
      if (*p >= L'A' && *p <= L'Z') *p = *p - L'A' + L'a';
    }

    if (StrCmp(Cmd, L"exit") == 0) {
      break;
    } else if (StrCmp(Cmd, L"help") == 0) {
      PrintHelp();
    } else if (StrCmp(Cmd, L"pwd") == 0) {
      DoCmdPwd();
    } else if (StrCmp(Cmd, L"dir") == 0 || StrCmp(Cmd, L"ls") == 0) {
      DoCmdDir(Arg);
    } else if (StrCmp(Cmd, L"cd") == 0) {
      DoCmdCd(Arg);
    } else if (StrCmp(Cmd, L"loadshell") == 0) {
      DoCmdLoadShell(Arg);
    } else if (StrCmp(Cmd, L"loadimg") == 0) {
      DoCmdLoadImg(Arg);
    } else if (StrCmp(Cmd, L"listapps") == 0) {
      DoCmdListApps();
    } else if (StrCmp(Cmd, L"gop") == 0) {
      DoGopCommand(Arg);
	} else if (StrCmp(Cmd, L"bgtext") == 0) {
		DoBgText(Arg);
    } else if (StrCmp(Cmd, L"setbuf") == 0) {
      if (Arg) {
        UINTN n = StrToUintn(Arg);
        SetTextBufferSize(n ? n : TEXT_BUFFER_DEFAULT_LINES);
      } else {
        TerminalPrintf(L"setbuf: missing number\n");
      }
    } else {
      TerminalPrintf(L"Unknown command '%s' - try 'help'\n", Cmd);
    }
  }

  // cleanup
  CloseCurrentDir();
  if (gCurPath) { FreeStr(gCurPath); gCurPath = NULL; }
  TerminalPrintf(L"HelloWorld: exiting terminal. Goodbye.\n");

  FreeTextBuffer();
  return EFI_SUCCESS;
}

/* ---------------------- text buffer implementation ---------------------- */

STATIC EFI_STATUS InitTextBuffer(UINTN Lines) {
  if (Lines == 0) Lines = TEXT_BUFFER_DEFAULT_LINES;
  TextBuffer = AllocateZeroPool(sizeof(CHAR16*) * Lines);
  if (!TextBuffer) return EFI_OUT_OF_RESOURCES;
  TextBufferLines = Lines;
  TextBufferHead = 0;
  TextBufferCount = 0;
  ViewOffset = 0;
  return EFI_SUCCESS;
}

STATIC VOID FreeTextBuffer(VOID) {
  if (TextBuffer) {
    for (UINTN i = 0; i < TextBufferCount; ++i) {
      UINTN idx = (TextBufferHead + i) % TextBufferLines;
      if (TextBuffer[idx]) { FreePool(TextBuffer[idx]); TextBuffer[idx] = NULL; }
    }
    FreePool(TextBuffer);
    TextBuffer = NULL;
  }
  TextBufferLines = 0;
  TextBufferHead = 0;
  TextBufferCount = 0;
  ViewOffset = 0;
}

/* Add a single line into the ring buffer (line is duplicated) */
STATIC VOID TerminalAddLine(CONST CHAR16 *Line) {
  if (!TextBuffer) return;
  CHAR16 *Dup = DupStr(Line ? Line : L"");
  if (!Dup) return;

  if (TextBufferCount < TextBufferLines) {
    UINTN idx = (TextBufferHead + TextBufferCount) % TextBufferLines;
    TextBuffer[idx] = Dup;
    TextBufferCount++;
  } else {
    // overwrite oldest
    if (TextBuffer[TextBufferHead]) FreePool(TextBuffer[TextBufferHead]);
    TextBuffer[TextBufferHead] = Dup;
    TextBufferHead = (TextBufferHead + 1) % TextBufferLines;
  }
}

/* Printf wrapper that also stores into buffer and prints if live */
STATIC VOID TerminalPrintf(CONST CHAR16 *Fmt, ...)
{
  CHAR16 Line[TEXT_LINE_MAX_CHARS];
  VA_LIST Args;
  VA_START(Args, Fmt);
  UnicodeVSPrint(Line, sizeof(Line), Fmt, Args);
  VA_END(Args);

  // store in buffer (split lines at '\n' boundaries)
  CHAR16 *p = Line;
  CHAR16 *segmentStart = p;
  while (*p) {
    if (*p == L'\n' || *p == L'\r') {
      *p = L'\0';
      TerminalAddLine(segmentStart);
      // advance past continuous newlines
      p++;
      while (*p == L'\n' || *p == L'\r') p++;
      segmentStart = p;
    } else {
      p++;
    }
  }
  if (segmentStart && *segmentStart) {
    TerminalAddLine(segmentStart);
  }

  // if view offset is 0 (live mode) print to screen immediately
  if (ViewOffset == 0) {
    // print the whole original line(s) as the user expects immediate output
    // We reprint the original with newlines if present.
    // For simplicity use Print because it's easiest to get on-screen quickly.
    // Note: splitting above removed \n; we print line and newline
    p = Line;
    CHAR16 *q = p;
    while (*q) {
      if (*q == L'\n' || *q == L'\r') { *q = L'\0'; }
      q++;
    }
    // Print once (if we had multiple segments they were stored separately)
    Print(L"%s\n", Line);
  } else {
    // When user is scrolled back, do not auto-print; the buffer retains the line
  }
}

/* Refresh whole screen using buffer content at current ViewOffset.
   ViewOffset==0 => show newest lines fitting screen.
   ViewOffset>0  => show older lines starting from tail-ViewOffset-screenRows+1
*/
STATIC VOID RefreshScreenWithBuffer(VOID) {
  UINTN screenRows = 25; // fallback
  EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
  if (!EFI_ERROR(GetGop(&Gop))) {
    UINTN vres = Gop->Mode->Info->VerticalResolution;
    // approximate char height => rows
    screenRows = (vres / CHAR_HEIGHT_ESTIMATE);
    if (screenRows < 5) screenRows = 5;
  } else {
    // try simple text mode heuristic: 25 rows default
    screenRows = 25;
  }

  // compute starting index in buffer
  if (TextBufferCount == 0) {
    // clear screen and return
    gST->ConOut->ClearScreen(gST->ConOut);
    return;
  }

  // Determine tail index (most recent element index)
  UINTN tailIndex = (TextBufferHead + TextBufferCount - 1) % TextBufferLines;
  INTN startOffsetFromTail = (INTN)ViewOffset + (INTN)screenRows - 1;
  INTN startIndexFromTail = (INTN)tailIndex - startOffsetFromTail;
  // clamp
  if (startIndexFromTail < 0) startIndexFromTail = 0;
  UINTN startIndex = (TextBufferHead + (UINTN)startIndexFromTail) % TextBufferLines;

  // Clear and print visible lines
  gST->ConOut->ClearScreen(gST->ConOut);
  UINTN printed = 0;
  UINTN idx = startIndex;
  while (printed < screenRows && printed < TextBufferCount) {
    if (TextBuffer[idx]) {
      Print(L"%s\r\n", TextBuffer[idx]);
    } else {
      Print(L"\r\n");
    }
    idx = (idx + 1) % TextBufferLines;
    printed++;
  }
}

/* Scroll helpers (lines param) */
STATIC VOID ScrollBufferUp(UINTN lines) {
  // Increase ViewOffset but not beyond buffer
  UINTN maxOffset = (TextBufferCount > 0) ? (TextBufferCount - 1) : 0;
  ViewOffset += (INTN)lines;
  if ((UINTN)ViewOffset > maxOffset) ViewOffset = (INTN)maxOffset;
  RefreshScreenWithBuffer();
}

STATIC VOID ScrollBufferDown(UINTN lines) {
  if (ViewOffset == 0) return;
  if ((UINTN)lines >= (UINTN)ViewOffset) {
    ViewOffset = 0;
  } else {
    ViewOffset -= (INTN)lines;
  }
  RefreshScreenWithBuffer();
}

/* Resize buffer */
STATIC VOID SetTextBufferSize(UINTN lines) {
  if (lines < 10) lines = 10;
  if (lines == TextBufferLines) {
    TerminalPrintf(L"setbuf: buffer size already %u\n", (UINT32)lines);
    return;
  }

  // Create new buffer
  CHAR16 **NewBuf = AllocateZeroPool(sizeof(CHAR16*) * lines);
  if (!NewBuf) {
    TerminalPrintf(L"setbuf: out of memory\n");
    return;
  }

  // copy most recent lines (up to lines)
  UINTN toCopy = (TextBufferCount < lines) ? TextBufferCount : lines;
  // start copying from (tail - toCopy + 1)
  for (UINTN i = 0; i < toCopy; ++i) {
    // element index from oldest for copyStart
    UINTN srcIdx = (TextBufferHead + TextBufferCount - toCopy + i) % TextBufferLines;
    NewBuf[i] = TextBuffer[srcIdx] ? DupStr(TextBuffer[srcIdx]) : NULL;
  }

  // free old buffer contents
  FreeTextBuffer();

  // adopt new
  TextBuffer = NewBuf;
  TextBufferLines = lines;
  TextBufferHead = 0;
  TextBufferCount = toCopy;
  ViewOffset = 0;
  TerminalPrintf(L"setbuf: buffer resized to %u lines\n", (UINT32)lines);
}

/* ---------------------- core helpers ---------------------- */

STATIC CHAR16 * DupStr(CONST CHAR16 *S) {
  if (S == NULL) return NULL;
  UINTN Len = StrLen(S) + 1;
  CHAR16 *D = AllocateZeroPool(Len * sizeof(CHAR16));
  if (D) {
    StrCpyS(D, Len, S);
  }
  return D;
}

STATIC VOID FreeStr(CHAR16 *S) {
  if (S) FreePool(S);
}

STATIC VOID CloseCurrentDir(VOID) {
  if (gCurDir && gCurDir != gRoot) {
    gCurDir->Close(gCurDir);
  }
  gCurDir = gRoot;
}

STATIC VOID PrintPrompt(VOID) {
  // always print prompt even when scrolled - user will see prompt, though older buffer shown
  if (ViewOffset != 0) {
    // indicate scroll mode
    Print(L"[SCROLL %d] %s> ", (INT32)ViewOffset, gCurPath ? gCurPath : L"\\");
  } else {
    Print(L"%s> ", gCurPath ? gCurPath : L"\\");
  }
}

/* Read line - with PageUp/PageDown support */
STATIC VOID ReadLine(CHAR16 *Buffer, UINTN BufferSize) {
  EFI_INPUT_KEY Key;
  UINTN Pos = 0;
  Buffer[0] = L'\0';

  for (;;) {
    // Wait for key
    while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_NOT_READY) {
      // spin
    }

    // handle page up / down
    if (Key.ScanCode == SCAN_PAGE_UP) {
      // scroll up by one screen (approx)
      ScrollBufferUp(20);
      // reprint prompt + current typed string
      PrintPrompt();
      if (Pos > 0) Print(L"%s", Buffer);
      continue;
    }
    if (Key.ScanCode == SCAN_PAGE_DOWN) {
      ScrollBufferDown(20);
      PrintPrompt();
      if (Pos > 0) Print(L"%s", Buffer);
      continue;
    }

    if (Key.UnicodeChar == CHAR_CARRIAGE_RETURN) {
      Print(L"\r\n");
      Buffer[Pos] = L'\0';
      return;
    } else if (Key.UnicodeChar == CHAR_BACKSPACE) {
      if (Pos > 0) {
        Pos--;
        Buffer[Pos] = L'\0';
        Print(L"\b \b");
      }
    } else if (Key.UnicodeChar >= 32) {
      if (Pos + 1 < BufferSize) {
        Buffer[Pos++] = Key.UnicodeChar;
        Buffer[Pos] = L'\0';
        Print(L"%c", Key.UnicodeChar);
      }
    }
    // ignore other keys
  }
}

/* Build absolute path: if Arg begins with '\' use Arg; otherwise append to gCurPath */
STATIC CHAR16 * BuildAbsolutePath(CONST CHAR16 *Arg) {
  if (Arg == NULL) return NULL;
  if (Arg[0] == L'\\') return DupStr(Arg);

  UINTN baseLen = gCurPath ? StrLen(gCurPath) : 0;
  BOOLEAN baseEndsWithSlash = (baseLen > 0 && gCurPath[baseLen - 1] == L'\\');
  UINTN argLen = StrLen(Arg);
  UINTN newLen = baseLen + (baseEndsWithSlash ? 0 : 1) + argLen + 1;
  CHAR16 *Buf = AllocateZeroPool(newLen * sizeof(CHAR16));
  if (!Buf) return NULL;
  StrCpyS(Buf, newLen, gCurPath ? gCurPath : L"\\");
  if (!baseEndsWithSlash) StrCatS(Buf, newLen, L"\\");
  StrCatS(Buf, newLen, Arg);
  return Buf;
}

/* OpenPathAsFile: same as in earlier code; StartDir is pointer to dir for relative paths */
STATIC EFI_STATUS OpenPathAsFile(EFI_FILE_PROTOCOL **StartDir, CONST CHAR16 *Path, EFI_FILE_PROTOCOL **OutFile, BOOLEAN MustBeDir) {
  if (StartDir == NULL || *StartDir == NULL || OutFile == NULL) return EFI_INVALID_PARAMETER;

  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *Base = NULL;
  CHAR16 *TryPath = NULL;
  CONST CHAR16 *OpenPath;

  if (Path == NULL || StrLen(Path) == 0) {
    *OutFile = *StartDir;
    return EFI_SUCCESS;
  }

  if (Path[0] == L'\\') {
    Base = gRoot;
    if (StrLen(Path) > 1) {
      TryPath = DupStr(Path + 1);
      OpenPath = TryPath;
    } else {
      OpenPath = L"";
    }
  } else {
    Base = *StartDir;
    TryPath = NULL;
    OpenPath = Path;
  }

  if (OpenPath[0] == L'\0') {
    *OutFile = Base;
    if (TryPath) FreeStr(TryPath);
    return EFI_SUCCESS;
  }

  Status = Base->Open(Base, OutFile, (CHAR16*)OpenPath, EFI_FILE_MODE_READ, 0);
  if (TryPath) FreeStr(TryPath);
  if (EFI_ERROR(Status)) {
    return Status;
  }

  if (MustBeDir) {
    UINTN InfoSize = SIZE_OF_EFI_FILE_INFO + 512 * sizeof(CHAR16);
    EFI_FILE_INFO *Info = AllocateZeroPool(InfoSize);
    if (Info == NULL) {
      (*OutFile)->Close(*OutFile);
      return EFI_OUT_OF_RESOURCES;
    }
    Status = (*OutFile)->GetInfo(*OutFile, &gEfiFileInfoGuid, &InfoSize, Info);
    if (EFI_ERROR(Status)) {
      FreePool(Info);
      (*OutFile)->Close(*OutFile);
      return Status;
    }
    if (!(Info->Attribute & EFI_FILE_DIRECTORY)) {
      FreePool(Info);
      (*OutFile)->Close(*OutFile);
      return EFI_UNSUPPORTED;
    }
    FreePool(Info);
  }

  return EFI_SUCCESS;
}

/* ---------------------- commands ---------------------- */

STATIC VOID DoCmdDir(CONST CHAR16 *Arg) {
  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *DirToRead = NULL;
  BOOLEAN LocalOpen = FALSE;

  if (Arg && StrLen(Arg) > 0) {
    Status = OpenPathAsFile(&gCurDir, Arg, &DirToRead, TRUE);
    if (EFI_ERROR(Status)) {
      TerminalPrintf(L"dir: cannot open '%s': %r\n", Arg, Status);
      return;
    }
    if (DirToRead != gCurDir && DirToRead != gRoot) LocalOpen = TRUE;
  } else {
    DirToRead = gCurDir;
  }

  if (DirToRead && DirToRead->SetPosition) DirToRead->SetPosition(DirToRead, 0);

  UINTN BufferSize = FILE_READ_BUFSIZE;
  VOID *Buffer = AllocateZeroPool(BufferSize);
  if (Buffer == NULL) {
    TerminalPrintf(L"dir: out of memory\n");
    if (LocalOpen && DirToRead) DirToRead->Close(DirToRead);
    return;
  }

  for (;;) {
    ZeroMem(Buffer, BufferSize);
    UINTN ReadSize = BufferSize;
    Status = DirToRead->Read(DirToRead, &ReadSize, Buffer);
    if (Status == EFI_BUFFER_TOO_SMALL) {
      FreePool(Buffer);
      BufferSize *= 2;
      Buffer = AllocateZeroPool(BufferSize);
      if (Buffer == NULL) {
        TerminalPrintf(L"dir: out of memory while realloc\n");
        break;
      }
      continue;
    }
    if (EFI_ERROR(Status)) {
      TerminalPrintf(L"dir: Read error: %r\n", Status);
      break;
    }
    if (ReadSize == 0) break;
    EFI_FILE_INFO *Info = (EFI_FILE_INFO*)Buffer;
    CHAR16 TypeChar = (Info->Attribute & EFI_FILE_DIRECTORY) ? L'D' : L'F';
    TerminalPrintf(L"%c %10llu  %s", TypeChar, (unsigned long long)Info->FileSize, Info->FileName);
  }

  FreePool(Buffer);
  if (LocalOpen && DirToRead) DirToRead->Close(DirToRead);
}

STATIC VOID DoCmdPwd(VOID) {
  TerminalPrintf(L"%s\n", gCurPath ? gCurPath : L"\\");
}

STATIC VOID DoCmdCd(CONST CHAR16 *Arg) {
  if (Arg == NULL || StrLen(Arg) == 0) {
    TerminalPrintf(L"cd: missing argument\n");
    return;
  }

  if (StrCmp(Arg, L"..") == 0) {
    // primitive parent handling
    if (gCurPath && StrLen(gCurPath) > 1) {
      UINTN len = StrLen(gCurPath);
      if (gCurPath[len - 1] == L'\\' && len > 1) {
        gCurPath[len - 1] = L'\0';
        len--;
      }
      CHAR16 *p = gCurPath + len - 1;
      while (p > gCurPath && *p != L'\\') p--;
      if (p == gCurPath) {
        FreeStr(gCurPath);
        gCurPath = DupStr(L"\\");
      } else {
        *p = L'\0';
      }
    }
    EFI_FILE_PROTOCOL *NewDir = NULL;
    EFI_STATUS st = OpenPathAsFile(&gRoot, gCurPath, &NewDir, TRUE);
    if (!EFI_ERROR(st) && NewDir) {
      if (gCurDir != gRoot && gCurDir != NewDir) gCurDir->Close(gCurDir);
      gCurDir = (NewDir == gRoot) ? gRoot : NewDir;
    }
    TerminalPrintf(L"cd: now %s\n", gCurPath);
    return;
  }

  EFI_FILE_PROTOCOL *NewDir = NULL;
  EFI_STATUS Status = OpenPathAsFile(&gCurDir, Arg, &NewDir, TRUE);
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"cd: cannot open directory '%s': %r\n", Arg, Status);
    return;
  }

  if (gCurDir && gCurDir != gRoot) gCurDir->Close(gCurDir);
  gCurDir = (NewDir == gRoot) ? gRoot : NewDir;

  CHAR16 *NewAbs = BuildAbsolutePath(Arg);
  if (NewAbs) {
    FreeStr(gCurPath);
    gCurPath = NewAbs;
  } else {
    TerminalPrintf(L"cd: warning: cannot update path string\n");
  }
  TerminalPrintf(L"cd: now %s\n", gCurPath);
}

//
// loadshell: previous behaviour kept but now works with Arg parameter (path or name).
// This code was extended in DoCmdLoadImg() which is the general image loader.
// If you call loadshell it will call loadimg with default path.
//
STATIC VOID DoCmdLoadShell(CONST CHAR16 *Arg) {
  // default: \EFI\Boot\shell.efi if Arg==NULL
  if (Arg == NULL) {
    DoCmdLoadImg(L"\\EFI\\Boot\\shell.efi");
  } else {
    DoCmdLoadImg(Arg);
  }
}

//
// loadimg <path-or-name>
// - if path starts with '\' it loads that absolute path
// - if name contains no backslash, it will be treated as \EFI\Boot\myApps\<name>
// - prompts user YES/NO before running
//
STATIC VOID DoCmdLoadImg(CONST CHAR16 *Arg) {
  if (Arg == NULL || StrLen(Arg) == 0) {
    TerminalPrintf(L"loadimg: missing argument (filename or absolute path)\n");
    return;
  }

  CHAR16 *AbsPath = NULL;
  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *FileHandle = NULL;
  BOOLEAN opened = FALSE;

  // decide path
  if (Arg[0] == L'\\') {
    AbsPath = DupStr(Arg);
  } else {
    // treat as name inside \EFI\Boot\myApps\<Arg>
    UINTN baseLen = StrLen(L"\\EFI\\Boot\\myApps\\");
    UINTN nameLen = StrLen(Arg);
    UINTN total = baseLen + nameLen + 1;
    AbsPath = AllocateZeroPool(total * sizeof(CHAR16));
    if (!AbsPath) {
      TerminalPrintf(L"loadimg: out of memory\n");
      return;
    }
    StrCpyS(AbsPath, total, L"\\EFI\\Boot\\myApps\\");
    StrCatS(AbsPath, total, Arg);
  }

  // try open to verify
  Status = OpenPathAsFile(&gCurDir, AbsPath, &FileHandle, FALSE);
  if (EFI_ERROR(Status) || FileHandle == NULL) {
    TerminalPrintf(L"loadimg: cannot find '%s': %r\n", AbsPath, Status);
    FreeStr(AbsPath);
    return;
  }
  opened = TRUE;

  TerminalPrintf(L"Found '%s'. Run image (YES/NO)? ", AbsPath);
  CHAR16 Answer[64];
  ReadLine(Answer, ARRAY_SIZE(Answer));
  for (CHAR16 *p = Answer; *p; ++p) if (*p >= L'A' && *p <= L'Z') *p = *p - L'A' + L'a';
  if (StrCmp(Answer, L"yes") != 0 && StrCmp(Answer, L"y") != 0) {
    TerminalPrintf(L"loadimg: aborted by user\n");
    if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
    FreeStr(AbsPath);
    return;
  }

  EFI_DEVICE_PATH_PROTOCOL *DevPath = FileDevicePath(mDeviceHandle, AbsPath);
  EFI_HANDLE ImgHandle = NULL;
  if (DevPath != NULL) {
    Status = gBS->LoadImage(FALSE, mImageHandle, DevPath, NULL, 0, &ImgHandle);
    if (EFI_ERROR(Status)) {
      TerminalPrintf(L"loadimg: LoadImage(device path) failed: %r\n", Status);
      FreePool(DevPath);
      if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
      FreeStr(AbsPath);
      return;
    }
    FreePool(DevPath);
  } else {
    // fallback - read file into memory and LoadImage from buffer
    UINTN InfoSize = SIZE_OF_EFI_FILE_INFO + 512 * sizeof(CHAR16);
    EFI_FILE_INFO *Info = AllocateZeroPool(InfoSize);
    if (!Info) {
      TerminalPrintf(L"loadimg: out of memory\n");
      if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
      FreeStr(AbsPath);
      return;
    }
    Status = FileHandle->GetInfo(FileHandle, &gEfiFileInfoGuid, &InfoSize, Info);
    if (EFI_ERROR(Status)) {
      TerminalPrintf(L"loadimg: GetInfo failed: %r\n", Status);
      FreePool(Info);
      if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
      FreeStr(AbsPath);
      return;
    }
    UINTN FileSize = (UINTN)Info->FileSize;
    FreePool(Info);

    VOID *ImageBuf = AllocateZeroPool(FileSize);
    if (!ImageBuf) {
      TerminalPrintf(L"loadimg: out of memory allocating %u bytes\n", FileSize);
      if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
      FreeStr(AbsPath);
      return;
    }

    UINTN ReadSize = FileSize;
    Status = FileHandle->Read(FileHandle, &ReadSize, ImageBuf);
    if (EFI_ERROR(Status) || ReadSize != FileSize) {
      TerminalPrintf(L"loadimg: file read failed: %r (read %u/%u)\n", Status, ReadSize, FileSize);
      FreePool(ImageBuf);
      if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
      FreeStr(AbsPath);
      return;
    }

    Status = gBS->LoadImage(FALSE, mImageHandle, NULL, ImageBuf, FileSize, &ImgHandle);
    if (EFI_ERROR(Status)) {
      TerminalPrintf(L"loadimg: LoadImage(buffer) failed: %r\n", Status);
      FreePool(ImageBuf);
      if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
      FreeStr(AbsPath);
      return;
    }
    // per spec, buffer can be freed after successful LoadImage
    FreePool(ImageBuf);
  }

  // Start the image
  TerminalPrintf(L"loadimg: starting image ...\n");
  UINTN ExitDataSize = 0;
  CHAR16 *ExitData = NULL;
  Status = gBS->StartImage(ImgHandle, &ExitDataSize, &ExitData);
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"loadimg: StartImage returned error: %r\n", Status);
  } else {
    TerminalPrintf(L"loadimg: image exited with status: %r\n", Status);
  }
  if (ExitData) {
    TerminalPrintf(L"loadimg: exit data (%u): %s\n", ExitDataSize, ExitData);
    FreePool(ExitData);
  }

  Status = gBS->UnloadImage(ImgHandle);
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"loadimg: UnloadImage returned: %r\n", Status);
  } else {
    TerminalPrintf(L"loadimg: image unloaded.\n");
  }

  if (opened && FileHandle && FileHandle != gRoot && FileHandle != gCurDir) FileHandle->Close(FileHandle);
  FreeStr(AbsPath);
}

STATIC VOID DoCmdListApps(VOID) {
  // list \EFI\Boot\myApps
  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *Dir = NULL;
  Status = OpenPathAsFile(&gRoot, L"\\EFI\\Boot\\myApps", &Dir, TRUE);
  if (EFI_ERROR(Status) || Dir == NULL) {
    TerminalPrintf(L"listapps: cannot open \\EFI\\Boot\\myApps : %r\n", Status);
    return;
  }

  if (Dir->SetPosition) Dir->SetPosition(Dir, 0);

  UINTN BufferSize = FILE_READ_BUFSIZE;
  VOID *Buffer = AllocateZeroPool(BufferSize);
  if (!Buffer) {
    TerminalPrintf(L"listapps: out of memory\n");
    return;
  }

  for (;;) {
    UINTN ReadSize = BufferSize;
    ZeroMem(Buffer, BufferSize);
    Status = Dir->Read(Dir, &ReadSize, Buffer);
    if (Status == EFI_BUFFER_TOO_SMALL) {
      FreePool(Buffer);
      BufferSize *= 2;
      Buffer = AllocateZeroPool(BufferSize);
      if (!Buffer) { TerminalPrintf(L"listapps: out of memory\n"); break; }
      continue;
    }
    if (EFI_ERROR(Status)) {
      TerminalPrintf(L"listapps: Read error: %r\n", Status);
      break;
    }
    if (ReadSize == 0) break;
    EFI_FILE_INFO *Info = (EFI_FILE_INFO*)Buffer;
    TerminalPrintf(L"%s %10llu  %s", (Info->Attribute & EFI_FILE_DIRECTORY) ? L"D" : L"F", (unsigned long long)Info->FileSize, Info->FileName);
  }

  FreePool(Buffer);
}

STATIC VOID PrintHelp(VOID) {
  TerminalPrintf(L"Commands:\n");
  TerminalPrintf(L"  help               - this text\n");
  TerminalPrintf(L"  dir [path]         - list directory contents (alias: ls)\n");
  TerminalPrintf(L"  pwd                - print current directory\n");
  TerminalPrintf(L"  cd <path>          - change directory (absolute: \\dir or relative)\n");
  TerminalPrintf(L"  loadshell [path]   - check and load \\EFI\\Boot\\shell.efi or provided path\n");
  TerminalPrintf(L"  loadimg <name|path>- load image; name => \\EFI\\Boot\\myApps\\<name>\n");
  TerminalPrintf(L"  listapps           - list files in \\EFI\\Boot\\myApps\\\n");
  TerminalPrintf(L"  gop list|set|bg    - GOP commands; bg sets whole-screen bg\n");
  TerminalPrintf(L"  bgtext <RRGGBB>    - set background rectangle behind text area\n");
  TerminalPrintf(L"  setbuf <lines>     - set text buffer lines\n");
  TerminalPrintf(L"  exit               - leave HelloWorld\n");
}

/* ---------------------- GOP helpers ---------------------- */

STATIC EFI_STATUS GetGop(EFI_GRAPHICS_OUTPUT_PROTOCOL **Out) {
  EFI_STATUS Status;
  EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop = NULL;
  Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID**)&Gop);
  if (EFI_ERROR(Status)) {
    return Status;
  }
  *Out = Gop;
  return EFI_SUCCESS;
}

STATIC VOID DoGopCommand(CONST CHAR16 *Arg) {
  if (Arg == NULL) {
    TerminalPrintf(L"gop: missing argument (list|set|bg)\n");
    return;
  }

  CHAR16 Work[LINE_BUFFER_SIZE];
  StrCpyS(Work, ARRAY_SIZE(Work), Arg);
  CHAR16 *Sub = Work;
  CHAR16 *Rest = StrStr(Sub, L" ");
  if (Rest) { *Rest = L'\0'; Rest++; while (*Rest == L' ') Rest++; }

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
      TerminalPrintf(L"  [%u] %ux%u, PixelFormat %u, PixelsPerScanLine %u", i, Info->HorizontalResolution, Info->VerticalResolution, Info->PixelFormat, Info->PixelsPerScanLine);
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
    
	/*
	CHAR8 asciiHex[16];
    UnicodeStrToAsciiStrS(Rest, asciiHex, ARRAY_SIZE(asciiHex));
    UINTN val = 0;
    for (UINTN i = 0; asciiHex[i] != '\0' && i < 8; ++i) {
      CHAR8 c = asciiHex[i]; UINTN d;
      if (c >= '0' && c <= '9') d = c - '0';
      else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
      else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
      else break;
      val = (val << 4) | d;
    }
	*/
	
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
  }

  TerminalPrintf(L"gop: unknown subcommand '%s'\n", Sub);
}

/* bgtext <RRGGBB> - fill rectangle behind the text area (approximate) */
STATIC VOID DoBgText(CONST CHAR16 *Arg) {
  if (Arg == NULL) { TerminalPrintf(L"bgtext: missing hex color (RRGGBB)\n"); return; }
  
  /*
  CHAR8 asciiHex[16];
  UnicodeStrToAsciiStrS(Arg, asciiHex, ARRAY_SIZE(asciiHex));
  UINTN val = 0;
  for (UINTN i = 0; asciiHex[i] != '\0' && i < 8; ++i) {
    CHAR8 c = asciiHex[i]; UINTN d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
    else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
    else break;
    val = (val << 4) | d;
  }
  */
  
  UINTN val = HexStrToUint(Arg);

  EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
  EFI_STATUS Status = GetGop(&Gop);
  if (EFI_ERROR(Status)) { TerminalPrintf(L"bgtext: GOP not available: %r\n", Status); return; }

  UINTN hres = Gop->Mode->Info->HorizontalResolution;
  UINTN vres = Gop->Mode->Info->VerticalResolution;
  UINTN rows = (vres / CHAR_HEIGHT_ESTIMATE);
  if (rows < 5) rows = 5;
  UINTN height = rows * CHAR_HEIGHT_ESTIMATE;
  if (height > vres) height = vres;

  EFI_GRAPHICS_OUTPUT_BLT_PIXEL Color;
  Color.Red = (UINT8)((val >> 16) & 0xFF);
  Color.Green = (UINT8)((val >> 8) & 0xFF);
  Color.Blue = (UINT8)(val & 0xFF);
  Color.Reserved = 0;

  Status = Gop->Blt(Gop, &Color, EfiBltVideoFill, 0, 0, 0, 0, hres, height, 0);
  if (EFI_ERROR(Status)) TerminalPrintf(L"bgtext: BLT failed: %r\n", Status);
  else TerminalPrintf(L"bgtext: text background filled with #%06x (height %u px)\n", val, (UINT32)height);
}

/* ---------------------- small utilities ---------------------- */

STATIC UINTN StrToUintn(CONST CHAR16 *Str) {
  if (Str == NULL) return 0;
  UINTN Val = 0;
  BOOLEAN Any = FALSE;
  while (*Str) {
    if (*Str >= L'0' && *Str <= L'9') {
      Any = TRUE;
      Val = Val * 10 + (UINTN)(*Str - L'0');
      Str++;
    } else break;
  }
  return Any ? Val : 0;
}
