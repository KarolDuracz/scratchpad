/** SimpleLogViewer.c
  Read-only text viewer for EDK2 UEFI demo.
  - list files in MYLOGS_REL_PATH
  - open by index
  - displays file contents (handles UTF-16LE BOM, UTF-8 BOM, ASCII)
  - PageUp / PageDown to scroll; Esc or Q to exit viewer
**/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/LoadedImage.h>
#include <Guid/FileInfo.h>

#define MYLOGS_REL_PATH L"\\EFI\\Boot\\myLogs"

#define CHAR_ESC       0x1B
#define QUIT_CHAR_LOWER L'q'
#define QUIT_CHAR_UPPER L'Q'

// Maximum files we'll list (demo)
#define MAX_FILES 256

// --- helpers to manage names ---
EFI_STATUS
ListFilesInDirectory(
  IN  EFI_FILE_PROTOCOL  *Root,
  IN  CHAR16             *DirPath,
  OUT CHAR16             ***OutNames,
  OUT UINTN              *OutCount
  )
{
  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *Dir = NULL;
  UINT8 *Buffer = NULL;
  UINTN BufferSize;
  UINTN Index = 0;
  CHAR16 **Names = NULL;

  Names = AllocateZeroPool(sizeof(CHAR16*) * MAX_FILES);
  if (Names == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }

  Status = Root->Open(Root, &Dir, DirPath, EFI_FILE_MODE_READ, 0);
  if (EFI_ERROR(Status)) {
    FreePool(Names);
    return Status;
  }

  BufferSize = 0x1000;
  Buffer = AllocatePool(BufferSize);
  if (Buffer == NULL) {
    Dir->Close(Dir);
    FreePool(Names);
    return EFI_OUT_OF_RESOURCES;
  }

  for (;;) {
    UINTN ReadSize = BufferSize;
    Status = Dir->Read(Dir, &ReadSize, Buffer);
    if (EFI_ERROR(Status)) break;
    if (ReadSize == 0) break; // end of dir

    EFI_FILE_INFO *FInfo = (EFI_FILE_INFO*)Buffer;
    if ((FInfo->Attribute & EFI_FILE_DIRECTORY) == 0) {
      if (Index < MAX_FILES) {
        UINTN NameSize = StrSize(FInfo->FileName);
        CHAR16 *NameDup = AllocatePool(NameSize);
        if (NameDup) {
          CopyMem(NameDup, FInfo->FileName, NameSize);
          Names[Index++] = NameDup;
        }
      }
    }
    ZeroMem(Buffer, BufferSize);
  }

  FreePool(Buffer);
  Dir->Close(Dir);

  *OutNames = Names;
  *OutCount = Index;
  return EFI_SUCCESS;
}

VOID
FreeNameList(CHAR16 **Names, UINTN Count)
{
  if (Names == NULL) return;
  for (UINTN i = 0; i < Count; ++i) {
    if (Names[i]) FreePool(Names[i]);
  }
  FreePool(Names);
}

// --- key input for index selection ---
INTN
PromptIndexAndGetChoice()
{
  EFI_INPUT_KEY Key;
  CHAR16 Buffer[32];
  UINTN Pos = 0;

  Print(L"\r\nEnter index number and press Enter (or Esc to cancel): ");
  for (;;) {
    UINTN EventIndex;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &EventIndex);
    if (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_SUCCESS) {
      if (Key.UnicodeChar == CHAR_LINEFEED || Key.UnicodeChar == CHAR_CARRIAGE_RETURN) {
        Buffer[Pos] = L'\0';
        if (Pos == 0) return -1;
        return (INTN)StrDecimalToUintn(Buffer);
      } else if (Key.UnicodeChar == CHAR_BACKSPACE) {
        if (Pos > 0) {
          Pos--;
          Print(L"\b \b");
        }
      } else if (Key.UnicodeChar == CHAR_ESC) {
        return -1;
      } else {
        if (Pos + 1 < sizeof(Buffer)/sizeof(Buffer[0]) && Key.UnicodeChar >= L'0' && Key.UnicodeChar <= L'9') {
          Buffer[Pos++] = Key.UnicodeChar;
          Buffer[Pos] = L'\0';
          Print(L"%c", Key.UnicodeChar);
        }
      }
    }
  }
}

// --- simple converters ---
// duplicate an ASCII/byte slice into a null-terminated CHAR16 string
CHAR16 *
AsciiToUnicodeDup(IN CHAR8 *A, IN UINTN Len)
{
  CHAR16 *U = AllocatePool((Len + 1) * sizeof(CHAR16));
  if (!U) return NULL;
  for (UINTN i = 0; i < Len; ++i) U[i] = (CHAR16)A[i];
  U[Len] = L'\0';
  return U;
}

// duplicate a range of CHAR16 into a new buffer (for UTF-16LE)
CHAR16 *
Unicode16Dup(IN CHAR16 *Src, IN UINTN Count)
{
  CHAR16 *D = AllocatePool((Count + 1) * sizeof(CHAR16));
  if (!D) return NULL;
  for (UINTN i = 0; i < Count; ++i) D[i] = Src[i];
  D[Count] = L'\0';
  return D;
}

// --- read entire file into memory (bytes) ---
EFI_STATUS
OpenFileReadAll(
  IN EFI_FILE_PROTOCOL *Root,
  IN CHAR16            *FilePath,
  OUT CHAR8            **OutBuffer,
  OUT UINTN            *OutSize
  )
{
  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *File = NULL;
  EFI_FILE_INFO *Info = NULL;
  UINTN InfoSize = 0;
  CHAR8 *Buf = NULL;

  Status = Root->Open(Root, &File, FilePath, EFI_FILE_MODE_READ, 0);
  if (EFI_ERROR(Status)) return Status;

  InfoSize = SIZE_OF_EFI_FILE_INFO + 512;
  Info = AllocatePool(InfoSize);
  if (!Info) {
    File->Close(File);
    return EFI_OUT_OF_RESOURCES;
  }

  Status = File->GetInfo(File, &gEfiFileInfoGuid, &InfoSize, Info);
  if (EFI_ERROR(Status)) {
    FreePool(Info);
    File->Close(File);
    return Status;
  }

  if (Info->FileSize == 0) {
    *OutBuffer = NULL;
    *OutSize = 0;
    FreePool(Info);
    File->Close(File);
    return EFI_SUCCESS;
  }

  if (Info->FileSize > (UINT64)((UINTN)-1)) {
    FreePool(Info);
    File->Close(File);
    return EFI_BAD_BUFFER_SIZE;
  }

  Buf = AllocatePool((UINTN)Info->FileSize + 1);
  if (!Buf) {
    FreePool(Info);
    File->Close(File);
    return EFI_OUT_OF_RESOURCES;
  }

  UINTN ReadSize = (UINTN)Info->FileSize;
  Status = File->Read(File, &ReadSize, Buf);
  if (EFI_ERROR(Status)) {
    FreePool(Buf);
    FreePool(Info);
    File->Close(File);
    return Status;
  }

  // safe-terminate (as bytes)
  Buf[ReadSize] = '\0';

  *OutBuffer = Buf;
  *OutSize = ReadSize;

  FreePool(Info);
  File->Close(File);
  return EFI_SUCCESS;
}

// --- create array of CHAR16* lines from the file buffer ---
// Supports:
//  - UTF-16LE with BOM 0xFF 0xFE
//  - UTF-8 with BOM 0xEF 0xBB 0xBF (we treat ASCII portion simply)
//  - ASCII/no BOM
EFI_STATUS
FileBufferToLines(
  IN  CHAR8  *Buf,
  IN  UINTN  BufSize,
  OUT CHAR16 ***OutLines,
  OUT UINTN  *OutLineCount
  )
{
  if (Buf == NULL || BufSize == 0) {
    *OutLines = NULL;
    *OutLineCount = 0;
    return EFI_SUCCESS;
  }

  CHAR16 **Lines = NULL;
  UINTN MaxLines = BufSize / 2 + 4;
  Lines = AllocateZeroPool(sizeof(CHAR16*) * MaxLines);
  if (!Lines) return EFI_OUT_OF_RESOURCES;
  UINTN LineCount = 0;

  // Detect BOMs
  if (BufSize >= 2 && (UINT8)Buf[0] == 0xFF && (UINT8)Buf[1] == 0xFE) {
    // UTF-16LE BOM -> interpret as CHAR16[]
    CHAR16 *U16 = (CHAR16*)(Buf + 2);
    UINTN U16Count = (BufSize - 2) / 2;
    UINTN pos = 0;
    while (pos < U16Count) {
      UINTN start = pos;
      while (pos < U16Count && U16[pos] != L'\n') pos++;
      // compute length excluding trailing CR
      UINTN len = pos - start;
      if (len > 0 && U16[start + len - 1] == L'\r') len--;
      CHAR16 *line = Unicode16Dup(&U16[start], len);
      if (!line) {
        for (UINTN i = 0; i < LineCount; ++i) FreePool(Lines[i]);
        FreePool(Lines);
        return EFI_OUT_OF_RESOURCES;
      }
      Lines[LineCount++] = line;
      if (pos < U16Count && U16[pos] == L'\n') pos++; // skip newline
    }
  } else {
    // UTF-8 BOM?: 0xEF 0xBB 0xBF
    UINTN offset = 0;
    if (BufSize >= 3 &&
        (UINT8)Buf[0] == 0xEF && (UINT8)Buf[1] == 0xBB && (UINT8)Buf[2] == 0xBF) {
      offset = 3;
    }
    // treat remaining as bytes; for this demo we assume either ASCII or UTF-8 with primarily ASCII content
    CHAR8 *p = Buf + offset;
    UINTN remaining = BufSize > offset ? BufSize - offset : 0;
    while (remaining > 0) {
      UINTN len = 0;
      while (len < remaining && p[len] != '\n') len++;
      UINTN actualLen = len;
      if (actualLen > 0 && p[actualLen - 1] == '\r') actualLen--;
      CHAR16 *u = AsciiToUnicodeDup(p, actualLen);
      if (!u) {
        for (UINTN i = 0; i < LineCount; ++i) FreePool(Lines[i]);
        FreePool(Lines);
        return EFI_OUT_OF_RESOURCES;
      }
      Lines[LineCount++] = u;
      if (len < remaining) {
        // skip newline
        p += (len + 1);
        remaining -= (len + 1);
      } else {
        // consumed remainder
        p += len;
        remaining -= len;
      }
    }
  }

  *OutLines = Lines;
  *OutLineCount = LineCount;
  return EFI_SUCCESS;
}

VOID
RenderWindowLines(CHAR16 **Lines, UINTN TotalLines, UINTN TopLine, UINTN WindowHeight, UINTN Columns)
{
  gST->ConOut->ClearScreen(gST->ConOut);

  for (UINTN i = 0; i < WindowHeight; ++i) {
    UINTN idx = TopLine + i;
    if (idx >= TotalLines) break;
    Print(L"%s\r\n", Lines[idx]);
  }
  Print(L"\r\n-- Page: %u/%u  (PageUp/PageDown to scroll, Esc/Q to exit) --\r\n",
        (UINT32)((TopLine/WindowHeight)+1),
        (UINT32)(((TotalLines + WindowHeight - 1) / WindowHeight)));
}

EFI_STATUS
ViewTextBufferWithPaging(CHAR8 *FileBuf, UINTN FileSize)
{
  if (FileBuf == NULL || FileSize == 0) {
    Print(L"\r\n[empty file]\r\nPress any key to return...\r\n");
    EFI_INPUT_KEY Key;
    UINTN EventIndex;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &EventIndex);
    gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
    return EFI_SUCCESS;
  }

  CHAR16 **Lines = NULL;
  UINTN LineCount = 0;
  EFI_STATUS Status = FileBufferToLines(FileBuf, FileSize, &Lines, &LineCount);
  if (EFI_ERROR(Status)) return Status;

  // If no lines parsed, create an empty indicator
  if (LineCount == 0) {
    Lines = AllocateZeroPool(sizeof(CHAR16*));
    Lines[0] = AllocatePool(4 * sizeof(CHAR16));
    if (Lines[0]) {
      StrCpyS(Lines[0], 4, L"[no text]");
      LineCount = 1;
    }
  }

  UINTN Columns = 80, Rows = 25;
  Status = gST->ConOut->QueryMode(gST->ConOut, gST->ConOut->Mode->Mode, &Columns, &Rows);
  if (EFI_ERROR(Status)) {
    Columns = 80;
    Rows = 25;
  }

  UINTN WindowHeight = (Rows > 3) ? (Rows - 3) : (Rows - 1);
  UINTN TopLine = 0;

  RenderWindowLines(Lines, LineCount, TopLine, WindowHeight, Columns);

  for (;;) {
    UINTN EventIndex;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &EventIndex);
    EFI_INPUT_KEY Key;
    if (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) != EFI_SUCCESS) continue;

    // Accept Esc (Unicode 0x1B) or q/Q to quit
    if (Key.UnicodeChar == CHAR_ESC || Key.UnicodeChar == QUIT_CHAR_LOWER || Key.UnicodeChar == QUIT_CHAR_UPPER) {
      break;
    } else if (Key.ScanCode == SCAN_PAGE_UP) {
      if (TopLine >= WindowHeight) TopLine -= WindowHeight;
      else TopLine = 0;
      RenderWindowLines(Lines, LineCount, TopLine, WindowHeight, Columns);
    } else if (Key.ScanCode == SCAN_PAGE_DOWN) {
      if (TopLine + WindowHeight < LineCount) {
        TopLine += WindowHeight;
        if (TopLine + WindowHeight > LineCount) {
          if (LineCount > WindowHeight) TopLine = LineCount - WindowHeight;
        }
      }
      RenderWindowLines(Lines, LineCount, TopLine, WindowHeight, Columns);
    }
  }

  for (UINTN i=0;i<LineCount;i++) FreePool(Lines[i]);
  FreePool(Lines);
  return EFI_SUCCESS;
}

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;
  EFI_LOADED_IMAGE_PROTOCOL *LoadedImage = NULL;
  EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *SimpleFs = NULL;
  EFI_FILE_PROTOCOL *Root = NULL;

  Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to get LoadedImage protocol: %r\r\n", Status);
    return Status;
  }

  Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&SimpleFs);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to get SimpleFileSystem protocol from device handle: %r\r\n", Status);
    return Status;
  }

  Status = SimpleFs->OpenVolume(SimpleFs, &Root);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to open volume: %r\r\n", Status);
    return Status;
  }

  CHAR16 **Names = NULL;
  UINTN Count = 0;
  Status = ListFilesInDirectory(Root, MYLOGS_REL_PATH, &Names, &Count);
  if (EFI_ERROR(Status)) {
    Print(L"Could not open directory %s : %r\r\n", MYLOGS_REL_PATH, Status);
    return Status;
  }

  if (Count == 0) {
    Print(L"No files found in %s\r\n", MYLOGS_REL_PATH);
    FreeNameList(Names, Count);
    return EFI_SUCCESS;
  }

  Print(L"\r\nFiles in %s:\r\n", MYLOGS_REL_PATH);
  for (UINTN i = 0; i < Count; ++i) {
    Print(L"  [%u]  %s\r\n", (UINT32)i, Names[i]);
  }

  INTN Choice = PromptIndexAndGetChoice();
  if (Choice < 0 || (UINTN)Choice >= Count) {
    Print(L"\r\nCanceled or invalid index.\r\n");
    FreeNameList(Names, Count);
    return EFI_SUCCESS;
  }

  UINTN PathLen = StrLen(MYLOGS_REL_PATH) + 1 + StrLen(Names[Choice]) + 1;
  CHAR16 *FullPath = AllocatePool(PathLen * sizeof(CHAR16));
  if (!FullPath) {
    FreeNameList(Names, Count);
    return EFI_OUT_OF_RESOURCES;
  }
  CopyMem(FullPath, MYLOGS_REL_PATH, StrSize(MYLOGS_REL_PATH));
  UINTN dirlen = StrLen(FullPath);
  if (FullPath[dirlen - 1] != L'\\') {
    FullPath[dirlen] = L'\\';
    FullPath[dirlen + 1] = L'\0';
    dirlen++;
  }
  StrCatS(FullPath, PathLen, Names[Choice]);

  CHAR8 *FileBuf = NULL;
  UINTN FileSize = 0;
  Status = OpenFileReadAll(Root, FullPath, &FileBuf, &FileSize);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to open or read file %s : %r\r\n", FullPath, Status);
    FreePool(FullPath);
    FreeNameList(Names, Count);
    return Status;
  }

  Print(L"\r\nOpening: %s (size: %u bytes)\r\n", FullPath, (UINT32)FileSize);
  Print(L"Press any key to start viewer (Esc or Q to quit)...\r\n");
  {
    EFI_INPUT_KEY K; UINTN E;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &E);
    gST->ConIn->ReadKeyStroke(gST->ConIn, &K);
  }

  Status = ViewTextBufferWithPaging(FileBuf, FileSize);

  if (FileBuf) FreePool(FileBuf);
  FreePool(FullPath);
  FreeNameList(Names, Count);

  Print(L"\r\nDone. Returning to shell.\r\n");
  return EFI_SUCCESS;
}



#if 0
/** SimpleLogViewer.c
  Simple EDK II UEFI demo: list files in MYLOGS_REL_PATH, pick by index,
  open and display file contents, scroll with PageUp/PageDown, exit with Esc.
**/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/LoadedImage.h>
#include <Guid/FileInfo.h>

#define MYLOGS_REL_PATH L"\\EFI\\Boot\\myLogs"
/* DO NOT redefine SCAN_PAGE_UP / SCAN_PAGE_DOWN here — they are defined by
   MdePkg's Include/Protocol/SimpleTextIn.h */

/* Use the charset macros provided by EDK: CHAR_LINEFEED, CHAR_CARRIAGE_RETURN, etc. */

#define CHAR_ESC       0x1B

// Maximum files we'll list (demo)
#define MAX_FILES 256

EFI_STATUS
ListFilesInDirectory(
  IN  EFI_FILE_PROTOCOL  *Root,
  IN  CHAR16             *DirPath,
  OUT CHAR16             ***OutNames,
  OUT UINTN              *OutCount
  )
{
  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *Dir = NULL;
  UINT8 *Buffer = NULL;
  UINTN BufferSize;
  UINTN Index = 0;
  CHAR16 **Names = NULL;

  Names = AllocateZeroPool(sizeof(CHAR16*) * MAX_FILES);
  if (Names == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }

  // Try to open directory (path relative to volume root)
  Status = Root->Open(Root, &Dir, DirPath, EFI_FILE_MODE_READ, 0);
  if (EFI_ERROR(Status)) {
    FreePool(Names);
    return Status;
  }

  // Allocate a read buffer large enough to hold directory entries
  BufferSize = 0x1000;
  Buffer = AllocatePool(BufferSize);
  if (Buffer == NULL) {
    Dir->Close(Dir);
    FreePool(Names);
    return EFI_OUT_OF_RESOURCES;
  }

  for (;;) {
    UINTN ReadSize = BufferSize;
    Status = Dir->Read(Dir, &ReadSize, Buffer);
    if (EFI_ERROR(Status)) break;
    if (ReadSize == 0) break; // end of dir

    EFI_FILE_INFO *FInfo = (EFI_FILE_INFO*)Buffer;
    // Skip directories; take only regular files
    if ((FInfo->Attribute & EFI_FILE_DIRECTORY) == 0) {
      if (Index < MAX_FILES) {
        UINTN NameSize = StrSize(FInfo->FileName);
        CHAR16 *NameDup = AllocatePool(NameSize);
        if (NameDup) {
          CopyMem(NameDup, FInfo->FileName, NameSize);
          Names[Index++] = NameDup;
        }
      }
    }
    // prepare for next entry
    ZeroMem(Buffer, BufferSize);
  }

  FreePool(Buffer);
  Dir->Close(Dir);

  *OutNames = Names;
  *OutCount = Index;
  return EFI_SUCCESS;
}

VOID
FreeNameList(CHAR16 **Names, UINTN Count)
{
  if (Names == NULL) return;
  for (UINTN i = 0; i < Count; ++i) {
    if (Names[i]) FreePool(Names[i]);
  }
  FreePool(Names);
}

INTN
PromptIndexAndGetChoice()
{
  EFI_INPUT_KEY Key;
  CHAR16 Buffer[32];
  UINTN Pos = 0;

  Print(L"\r\nEnter index number and press Enter (or Esc to cancel): ");
  for (;;) {
    UINTN EventIndex;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &EventIndex);
    if (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_SUCCESS) {
      if (Key.UnicodeChar == CHAR_LINEFEED || Key.UnicodeChar == CHAR_CARRIAGE_RETURN) {
        Buffer[Pos] = L'\0';
        if (Pos == 0) return -1;
        return (INTN)StrDecimalToUintn(Buffer);
      } else if (Key.UnicodeChar == CHAR_BACKSPACE) {
        if (Pos > 0) {
          Pos--;
          Print(L"\b \b");
        }
      } else if (Key.UnicodeChar == CHAR_ESC) {
        return -1;
      } else {
        if (Pos + 1 < sizeof(Buffer)/sizeof(Buffer[0]) && Key.UnicodeChar >= L'0' && Key.UnicodeChar <= L'9') {
          Buffer[Pos++] = Key.UnicodeChar;
          Buffer[Pos] = L'\0';
          Print(L"%c", Key.UnicodeChar);
        }
      }
    }
  }
}

CHAR16 *
AsciiToUnicodeDup(IN CHAR8 *A, IN UINTN Len)
{
  CHAR16 *U = AllocatePool((Len + 1) * sizeof(CHAR16));
  if (!U) return NULL;
  for (UINTN i = 0; i < Len; ++i) U[i] = (CHAR16)A[i];
  U[Len] = L'\0';
  return U;
}

EFI_STATUS
OpenFileReadAll(
  IN EFI_FILE_PROTOCOL *Root,
  IN CHAR16            *FilePath,
  OUT CHAR8            **OutBuffer,
  OUT UINTN            *OutSize
  )
{
  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *File = NULL;
  EFI_FILE_INFO *Info = NULL;
  UINTN InfoSize = 0;
  CHAR8 *Buf = NULL;

  Status = Root->Open(Root, &File, FilePath, EFI_FILE_MODE_READ, 0);
  if (EFI_ERROR(Status)) return Status;

  InfoSize = SIZE_OF_EFI_FILE_INFO + 512;
  Info = AllocatePool(InfoSize);
  if (!Info) {
    File->Close(File);
    return EFI_OUT_OF_RESOURCES;
  }

  Status = File->GetInfo(File, &gEfiFileInfoGuid, &InfoSize, Info);
  if (EFI_ERROR(Status)) {
    FreePool(Info);
    File->Close(File);
    return Status;
  }

  if (Info->FileSize == 0) {
    *OutBuffer = NULL;
    *OutSize = 0;
    FreePool(Info);
    File->Close(File);
    return EFI_SUCCESS;
  }

  // Ensure the file size fits in UINTN (important on 32-bit builds)
  if (Info->FileSize > (UINT64)((UINTN)-1)) {
    FreePool(Info);
    File->Close(File);
    return EFI_BAD_BUFFER_SIZE;
  }

  Buf = AllocatePool((UINTN)Info->FileSize + 1);
  if (!Buf) {
    FreePool(Info);
    File->Close(File);
    return EFI_OUT_OF_RESOURCES;
  }

  UINTN ReadSize = (UINTN)Info->FileSize;
  Status = File->Read(File, &ReadSize, Buf);
  if (EFI_ERROR(Status)) {
    FreePool(Buf);
    FreePool(Info);
    File->Close(File);
    return Status;
  }

  Buf[ReadSize] = '\0';

  *OutBuffer = Buf;
  *OutSize = ReadSize;

  FreePool(Info);
  File->Close(File);
  return EFI_SUCCESS;
}

VOID
RenderWindowLines(CHAR16 **Lines, UINTN TotalLines, UINTN TopLine, UINTN WindowHeight, UINTN Columns)
{
  gST->ConOut->ClearScreen(gST->ConOut);

  for (UINTN i = 0; i < WindowHeight; ++i) {
    UINTN idx = TopLine + i;
    if (idx >= TotalLines) break;
    Print(L"%s\r\n", Lines[idx]);
  }
  Print(L"\r\n-- Page: %u/%u  (PageUp/PageDown to scroll, Esc to exit) --\r\n",
        (UINT32)((TopLine/WindowHeight)+1),
        (UINT32)(((TotalLines + WindowHeight - 1) / WindowHeight)));
}

EFI_STATUS
ViewTextBufferWithPaging(CHAR8 *FileBuf, UINTN FileSize)
{
  if (FileBuf == NULL || FileSize == 0) {
    Print(L"\r\n[empty file]\r\nPress any key to return...\r\n");
    EFI_INPUT_KEY Key;
    UINTN EventIndex;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &EventIndex);
    gST->ConIn->ReadKeyStroke(gST->ConIn, &Key);
    return EFI_SUCCESS;
  }

  CHAR16 **Lines = AllocateZeroPool(sizeof(CHAR16*) * (FileSize / 10 + 4));
  if (!Lines) return EFI_OUT_OF_RESOURCES;
  UINTN LineCount = 0;

  CHAR8 *p = FileBuf;
  CHAR8 *start = p;
  UINTN remaining = FileSize;
  while (remaining > 0) {
    UINTN len = 0;
    while (len < remaining && p[len] != '\n') len++;
    UINTN actualLen = len;
    if (actualLen > 0 && p[actualLen - 1] == '\r') actualLen--;
    CHAR16 *u = AsciiToUnicodeDup(start, actualLen);
    if (!u) {
      for (UINTN i=0;i<LineCount;i++) FreePool(Lines[i]);
      FreePool(Lines);
      return EFI_OUT_OF_RESOURCES;
    }
    Lines[LineCount++] = u;

    if (len < remaining) {
      p += (len + 1);
      remaining -= (len + 1);
      start = p;
    } else {
      break;
    }
  }
  if (remaining > 0) {
    UINTN actualLen = remaining;
    if (actualLen > 0 && p[actualLen - 1] == '\r') actualLen--;
    CHAR16 *u = AsciiToUnicodeDup(p, actualLen);
    if (u) Lines[LineCount++] = u;
  }

  UINTN Columns = 80, Rows = 25;
  EFI_STATUS Status = gST->ConOut->QueryMode(gST->ConOut, gST->ConOut->Mode->Mode, &Columns, &Rows);
  if (EFI_ERROR(Status)) {
    Columns = 80;
    Rows = 25;
  }

  UINTN WindowHeight = (Rows > 3) ? (Rows - 3) : (Rows - 1);
  UINTN TopLine = 0;

  RenderWindowLines(Lines, LineCount, TopLine, WindowHeight, Columns);

  for (;;) {
    UINTN EventIndex;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &EventIndex);
    EFI_INPUT_KEY Key;
    if (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) != EFI_SUCCESS) continue;

    if (Key.UnicodeChar == CHAR_ESC) {
      break;
    } else if (Key.ScanCode == SCAN_PAGE_UP) {
      if (TopLine >= WindowHeight) TopLine -= WindowHeight;
      else TopLine = 0;
      RenderWindowLines(Lines, LineCount, TopLine, WindowHeight, Columns);
    } else if (Key.ScanCode == SCAN_PAGE_DOWN) {
      if (TopLine + WindowHeight < LineCount) {
        TopLine += WindowHeight;
        if (TopLine + WindowHeight > LineCount) {
          if (LineCount > WindowHeight) TopLine = LineCount - WindowHeight;
        }
      }
      RenderWindowLines(Lines, LineCount, TopLine, WindowHeight, Columns);
    }
  }

  for (UINTN i=0;i<LineCount;i++) FreePool(Lines[i]);
  FreePool(Lines);
  return EFI_SUCCESS;
}

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;
  EFI_LOADED_IMAGE_PROTOCOL *LoadedImage = NULL;
  EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *SimpleFs = NULL;
  EFI_FILE_PROTOCOL *Root = NULL;

  Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to get LoadedImage protocol: %r\r\n", Status);
    return Status;
  }

  Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&SimpleFs);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to get SimpleFileSystem protocol from device handle: %r\r\n", Status);
    return Status;
  }

  Status = SimpleFs->OpenVolume(SimpleFs, &Root);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to open volume: %r\r\n", Status);
    return Status;
  }

  CHAR16 **Names = NULL;
  UINTN Count = 0;
  Status = ListFilesInDirectory(Root, MYLOGS_REL_PATH, &Names, &Count);
  if (EFI_ERROR(Status)) {
    Print(L"Could not open directory %s : %r\r\n", MYLOGS_REL_PATH, Status);
    return Status;
  }

  if (Count == 0) {
    Print(L"No files found in %s\r\n", MYLOGS_REL_PATH);
    FreeNameList(Names, Count);
    return EFI_SUCCESS;
  }

  Print(L"\r\nFiles in %s:\r\n", MYLOGS_REL_PATH);
  for (UINTN i = 0; i < Count; ++i) {
    Print(L"  [%u]  %s\r\n", (UINT32)i, Names[i]);
  }

  INTN Choice = PromptIndexAndGetChoice();
  if (Choice < 0 || (UINTN)Choice >= Count) {
    Print(L"\r\nCanceled or invalid index.\r\n");
    FreeNameList(Names, Count);
    return EFI_SUCCESS;
  }

  UINTN PathLen = StrLen(MYLOGS_REL_PATH) + 1 + StrLen(Names[Choice]) + 1;
  CHAR16 *FullPath = AllocatePool(PathLen * sizeof(CHAR16));
  if (!FullPath) {
    FreeNameList(Names, Count);
    return EFI_OUT_OF_RESOURCES;
  }
  CopyMem(FullPath, MYLOGS_REL_PATH, StrSize(MYLOGS_REL_PATH));
  UINTN dirlen = StrLen(FullPath);
  if (FullPath[dirlen - 1] != L'\\') {
    FullPath[dirlen] = L'\\';
    FullPath[dirlen + 1] = L'\0';
    dirlen++;
  }
  StrCatS(FullPath, PathLen, Names[Choice]);

  CHAR8 *FileBuf = NULL;
  UINTN FileSize = 0;
  Status = OpenFileReadAll(Root, FullPath, &FileBuf, &FileSize);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to open or read file %s : %r\r\n", FullPath, Status);
    FreePool(FullPath);
    FreeNameList(Names, Count);
    return Status;
  }

  Print(L"\r\nOpening: %s (size: %u bytes)\r\n", FullPath, (UINT32)FileSize);
  Print(L"Press any key to start viewer...\r\n");
  {
    EFI_INPUT_KEY K; UINTN E;
    gBS->WaitForEvent(1, &gST->ConIn->WaitForKey, &E);
    gST->ConIn->ReadKeyStroke(gST->ConIn, &K);
  }

  Status = ViewTextBufferWithPaging(FileBuf, FileSize);

  if (FileBuf) FreePool(FileBuf);
  FreePool(FullPath);
  FreeNameList(Names, Count);

  Print(L"\r\nDone. Returning to shell.\r\n");
  return EFI_SUCCESS;
}
#endif
