#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DevicePathLib.h>
#include <Library/PrintLib.h>    /* UnicodeVSPrint */
#include <stdarg.h>             /* VA_LIST, VA_START, VA_END */
#include <Protocol/LoadedImage.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/GraphicsOutput.h>
#include <Protocol/SimpleTextIn.h>    // SCAN_PAGE_UP / SCAN_PAGE_DOWN
#include <Guid/FileInfo.h>
#include <Library/UefiApplicationEntryPoint.h>

#define LINE_BUFFER_SIZE 1024
#define FILE_READ_BUFSIZE 0x1000

// text buffer defaults //
#define TEXT_BUFFER_DEFAULT_LINES 400
#define TEXT_LINE_MAX_CHARS 1024
#define CHAR_HEIGHT_ESTIMATE 16 // pixels; approximate for "bgtext" height calculation

////////////////////////////////////////////////////////////////////////////////////////////////////////

	/* fixes for log duplication */

//#define __FIRST_WORKING_VERSION__

////////////////////////////////////////////////////////////////////////////////////////////////////////

/* ---------------------- secure logging helpers ----------------------
   Usage: call SaveConsoleBufferToMyLogs() to create a new unique file
   under \EFI\Boot\myLogs\ containing the current TextBuffer contents.
-------------------------------------------------------------------*/

#define MYLOGS_REL_PATH L"\\EFI\\Boot\\myLogs"
#define LOG_FILENAME_BASE L"log"
#define LOG_FILENAME_EXT L".txt"
#define LOG_FILENAME_MAX 128
#define RAM_TEST_BYTES (4 * 1024) /* quick memory allocation test */
STATIC UINTN gLogSequenceCounter = 0; /* used for filename uniqueness when necessary */

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
/* forward declaration for TerminalPrintf (no PRINTF_ATTR here) */
STATIC VOID TerminalPrintf(CONST CHAR16 *Fmt, ...);

STATIC VOID RefreshScreenWithBuffer(VOID);
STATIC VOID ScrollBufferUp(UINTN lines);
STATIC VOID ScrollBufferDown(UINTN lines);
STATIC VOID SetTextBufferSize(UINTN lines);

/* utility */
STATIC UINTN StrToUintn(CONST CHAR16 *Str);
STATIC VOID DoBgText(CONST CHAR16 *Arg);

// fixed log 
/* capture control - memory-backed capture */
STATIC VOID DoCmdCapMem(CONST CHAR16 *Arg);

/* diagnostic for bugs - duplicating lines in logs 01-11-2025 */
STATIC VOID DiagnosticCheckCaptureBuffers(VOID);

/* temp buffer used by SafeConsolePrint for formatting */
STATIC CHAR16 gTmpLineBuffer[LINE_BUFFER_SIZE];


/* wrapper prototype type for OutputString (already used) */
typedef
EFI_STATUS
(EFIAPI *EFI_TEXT_STRING) (
  IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL  *This,
  IN CHAR16                            *String
  );

/* existing wrapper state */
STATIC EFI_TEXT_STRING gOriginalOutputString = NULL;
STATIC BOOLEAN        gConsoleCaptureEnabled = FALSE;

/* memory-backed capture buffer */
STATIC CHAR16  *gCaptureMem = NULL;        /* contiguous block of CHAR16 for slots */
STATIC UINTN    gCaptureMemBytes = 0;
STATIC UINTN    gCaptureMaxLines = 0;
STATIC UINTN    gCaptureMaxChars = 0;      /* chars per slot (not bytes) */
STATIC UINTN    gCaptureSlotChars = 0;     /* = gCaptureMaxChars + 1 (for NUL) */
STATIC UINTN    gCaptureHead = 0;          /* next-slot index (circular) */
STATIC UINTN    gCaptureCount = 0;         /* number of occupied slots (<= maxlines) */
STATIC BOOLEAN  gCaptureMemActive = FALSE;

/* protocol-wide patching & notify */
STATIC EFI_STATUS PatchAllSimpleTextOuts(VOID);
STATIC VOID UnpatchAllSimpleTextOuts(VOID);
STATIC VOID EFIAPI SimpleTextOutNotifyCallback(IN EFI_EVENT Event, IN VOID *Context);
STATIC EFI_STATUS EnableAllTextOutPatch(VOID);
STATIC VOID DisableAllTextOutPatch(VOID);

/* registration event */
STATIC EFI_EVENT gSimpleTextOutNotifyEvent = NULL;
STATIC VOID    *gSimpleTextOutNotifyReg = NULL;

#define MAX_PATCHED_TEXTOUTS 128

#ifdef __FIRST_WORKING_VERSION__
typedef struct {
  EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *Proto;
  EFI_TEXT_STRING                 OriginalOutputString;
} PATCHED_TEXTOUT_ENTRY;

STATIC PATCHED_TEXTOUT_ENTRY gPatchedTextOuts[MAX_PATCHED_TEXTOUTS];
STATIC UINTN gPatchedCount = 0;
#else
#define ACCUM_CHARS_DEFAULT 2048

typedef struct {
  EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *Proto;
  EFI_TEXT_STRING                 OriginalOutputString;
  CHAR16                         *AccumBuf;      /* per-proto accumulator (NUL terminated) */
  UINTN                           AccumBufChars; /* capacity in CHAR16 */
} PATCHED_TEXTOUT_ENTRY;

STATIC PATCHED_TEXTOUT_ENTRY gPatchedTextOuts[MAX_PATCHED_TEXTOUTS];
STATIC UINTN gPatchedCount = 0;
#endif


/* simple whitelist for filename characters */
STATIC
BOOLEAN
IsValidLogFileNameChar(CHAR16 c)
{
  if ((c >= L'0' && c <= L'9') ||
      (c >= L'A' && c <= L'Z') ||
      (c >= L'a' && c <= L'z') ||
      c == L'-' || c == L'_' || c == L'.')
  {
    return TRUE;
  }
  return FALSE;
}

/* ensure myLogs directory exists; return open directory handle in OutDir
   - caller must Close() the returned handle if it's not gRoot.
*/
STATIC
EFI_STATUS
EnsureMyLogsDir(EFI_FILE_PROTOCOL **OutDir)
{
  if (OutDir == NULL) return EFI_INVALID_PARAMETER;

  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *Dir = NULL;

  /* try existing via helper (absolute path) */
  Status = OpenPathAsFile(&gRoot, MYLOGS_REL_PATH, &Dir, TRUE);
  if (!EFI_ERROR(Status) && Dir != NULL) {
    *OutDir = Dir;
    return EFI_SUCCESS;
  }

  /* If not found, attempt to create it under \EFI\Boot\ (demo already has \EFI\Boot) */
  /* Use gRoot->Open to create. Note: when using gRoot->Open with absolute path you must pass without leading '\' */
  Status = gRoot->Open(gRoot, &Dir, L"EFI\\Boot\\myLogs", EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, EFI_FILE_DIRECTORY);
  if (!EFI_ERROR(Status) && Dir != NULL) {
    *OutDir = Dir;
    return EFI_SUCCESS;
  }

  /* If creation failed, return original error */
  return Status;
}

/* quick RAM test: allocate Bytes, fill pattern, verify, free */
STATIC
EFI_STATUS
RamQuickTest(UINTN Bytes)
{
  if (Bytes == 0) return EFI_INVALID_PARAMETER;

  VOID *Buf = NULL;
  EFI_STATUS Status = EFI_SUCCESS;
  Buf = AllocatePool(Bytes);
  if (Buf == NULL) return EFI_OUT_OF_RESOURCES;

  /* write pattern */
  for (UINTN i = 0; i < Bytes; ++i) {
    ((UINT8*)Buf)[i] = (UINT8)(i & 0xFF);
  }
  /* verify */
  for (UINTN i = 0; i < Bytes; ++i) {
    if (((UINT8*)Buf)[i] != (UINT8)(i & 0xFF)) {
      Status = EFI_DEVICE_ERROR;
      break;
    }
  }

  FreePool(Buf);
  return Status;
}

/* build a safe filename into OutName; OutNameChars is max CHAR16 characters available.
   Format: log-YYYYMMDD-HHMMSS-XXXX.txt  (XXXX = sequence counter)
*/
STATIC
EFI_STATUS
BuildLogFileName(CHAR16 *OutName, UINTN OutNameChars)
{
  if (OutName == NULL || OutNameChars == 0) return EFI_INVALID_PARAMETER;

  EFI_TIME Time;
  EFI_STATUS Status = EFI_SUCCESS;

  /* Use gST->RuntimeServices->GetTime (gRT wasn't declared in this file) */
  if (gST && gST->RuntimeServices && gST->RuntimeServices->GetTime) {
    Status = gST->RuntimeServices->GetTime(&Time, NULL);
    if (EFI_ERROR(Status)) {
      /* fall back to a known default time if GetTime fails */
      Time.Year = 1970; Time.Month = 1; Time.Day = 1;
      Time.Hour = 0; Time.Minute = 0; Time.Second = 0;
    }
  } else {
    /* no runtime time available */
    Time.Year = 1970; Time.Month = 1; Time.Day = 1;
    Time.Hour = 0; Time.Minute = 0; Time.Second = 0;
    Status = EFI_UNSUPPORTED;
  }

  /* build filename */
  gLogSequenceCounter++; /* always increment to improve uniqueness */
  UnicodeSPrint(OutName, OutNameChars * sizeof(CHAR16),
                L"%s-%04u%02u%02u-%02u%02u%02u-%04u%s",
                LOG_FILENAME_BASE,
                (UINT32)Time.Year,
                (UINT32)Time.Month,
                (UINT32)Time.Day,
                (UINT32)Time.Hour,
                (UINT32)Time.Minute,
                (UINT32)Time.Second,
                (UINT32)(gLogSequenceCounter & 0xFFFF),
                LOG_FILENAME_EXT);

  /* sanitize: ensure only safe chars (keep '-' '_' '.' digits letters) */
  for (UINTN i = 0; i < StrLen(OutName); ++i) {
    if (!IsValidLogFileNameChar(OutName[i])) {
      /* replace unsafe with '_' */
      OutName[i] = L'_';
    }
  }

  return EFI_SUCCESS;
}


/* Create a unique file under Dir (which must point to \EFI\Boot\myLogs directory).
   OutFile receives the opened file handle (open for read/write, created). Caller must Close it.
   OutFinalName optionally receives the filename allocated by caller buffer (maxChars).
*/
STATIC
EFI_STATUS
CreateUniqueLogFile(EFI_FILE_PROTOCOL *Dir, EFI_FILE_PROTOCOL **OutFile, CHAR16 *OutFinalName, UINTN MaxChars)
{
  if (Dir == NULL || OutFile == NULL) return EFI_INVALID_PARAMETER;
  if (OutFinalName == NULL || MaxChars == 0) return EFI_INVALID_PARAMETER;

  EFI_STATUS Status;
  CHAR16 Candidate[LOG_FILENAME_MAX];
  UINTN tryCount = 0;

  do {
    Status = BuildLogFileName(Candidate, ARRAY_SIZE(Candidate));
    if (EFI_ERROR(Status)) return Status;

    /* check existence - try open read-only */
    EFI_FILE_PROTOCOL *Check = NULL;
    Status = Dir->Open(Dir, &Check, Candidate, EFI_FILE_MODE_READ, 0);
    if (!EFI_ERROR(Status) && Check != NULL) {
      /* file exists: close and try again */
      if (Check != gRoot && Check != gCurDir) Check->Close(Check);
      tryCount++;
      /* slightly change counter to attempt new name */
      gLogSequenceCounter++;
      continue;
    }

    /* not present - create new file safely */
    EFI_FILE_PROTOCOL *NewFile = NULL;
    Status = Dir->Open(Dir, &NewFile, Candidate, EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, EFI_FILE_ARCHIVE);
    if (EFI_ERROR(Status)) {
      /* cannot create - return error */
      return Status;
    }

    /* success */
    StrCpyS(OutFinalName, MaxChars, Candidate);
    *OutFile = NewFile;
    return EFI_SUCCESS;

  } while (tryCount < 1000);

  return EFI_LOAD_ERROR; /* too many collisions */
}

/* Helper: perform a small create/write/read/delete test inside LogsDir.
   This verifies the volume is responsive and the directory is operable.
   OutTestName optionally receives the temporary test filename (caller buffer).
*/
STATIC
EFI_STATUS
HealthCheckLogsDir(EFI_FILE_PROTOCOL *LogsDir, CHAR16 *OutTestName, UINTN OutChars)
{
  if (LogsDir == NULL) return EFI_INVALID_PARAMETER;

  EFI_STATUS Status;
  CHAR16 TestName[64];
  /* build a short test filename, keep it hidden-ish */
  gLogSequenceCounter++;
  UnicodeSPrint(TestName, sizeof(TestName), L".mylogs_test_%04u.tmp", (UINT32)(gLogSequenceCounter & 0xFFFF));

  if (OutTestName && OutChars > 0) {
    StrCpyS(OutTestName, OutChars, TestName);
  }

  /* Attempt to create the test file (relative open on LogsDir) */
  EFI_FILE_PROTOCOL *TestFile = NULL;
  Status = LogsDir->Open(LogsDir, &TestFile, TestName, EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, 0);
  if (EFI_ERROR(Status) || TestFile == NULL) {
    TerminalPrintf(L"HealthCheck: FAIL - cannot create test file '%s': %r\n", TestName, Status);
    return Status;
  }
  TerminalPrintf(L"HealthCheck: OK - test file '%s' created\n", TestName);

  /* Write a small pattern */
  UINT8 Pattern[2] = { 0x5A, 0xA5 }; /* arbitrary */
  UINTN WriteSize = sizeof(Pattern);
  Status = TestFile->Write(TestFile, &WriteSize, Pattern);
  if (EFI_ERROR(Status) || WriteSize != sizeof(Pattern)) {
    TerminalPrintf(L"HealthCheck: FAIL - write to test file failed: %r (wrote %u)\n", Status, (UINT32)WriteSize);
    /* attempt to delete partial file */
    EFI_STATUS d = TestFile->Delete(TestFile);
    if (EFI_ERROR(d)) { TestFile->Close(TestFile); }
    return EFI_DEVICE_ERROR;
  }
  TerminalPrintf(L"HealthCheck: OK - wrote %u bytes to test file\n", (UINT32)WriteSize);

  /* Seek back to beginning (SetPosition may be available) */
  if (TestFile->SetPosition) TestFile->SetPosition(TestFile, 0);

  /* Read back */
  UINT8 ReadBuf[2] = {0};
  UINTN ReadSize = sizeof(ReadBuf);
  Status = TestFile->Read(TestFile, &ReadSize, ReadBuf);
  if (EFI_ERROR(Status) || ReadSize != sizeof(ReadBuf)) {
    TerminalPrintf(L"HealthCheck: FAIL - read from test file failed: %r (read %u)\n", Status, (UINT32)ReadSize);
    EFI_STATUS d = TestFile->Delete(TestFile);
    if (EFI_ERROR(d)) { TestFile->Close(TestFile); }
    return EFI_DEVICE_ERROR;
  }

  /* Validate */
  if (ReadBuf[0] != Pattern[0] || ReadBuf[1] != Pattern[1]) {
    TerminalPrintf(L"HealthCheck: FAIL - pattern mismatch (wrote %02x%02x read %02x%02x)\n",
                   Pattern[0], Pattern[1], ReadBuf[0], ReadBuf[1]);
    EFI_STATUS d = TestFile->Delete(TestFile);
    if (EFI_ERROR(d)) { TestFile->Close(TestFile); }
    return EFI_COMPROMISED_DATA; /* generic error code */
  }
  TerminalPrintf(L"HealthCheck: OK - read/verify test succeeded\n");

  /* Delete test file (Delete closes the handle) */
  Status = TestFile->Delete(TestFile);
  if (EFI_ERROR(Status)) {
    /* If Delete fails, try to close as fallback */
    TerminalPrintf(L"HealthCheck: WARN - delete test file failed: %r (attempting Close)\n", Status);
    TestFile->Close(TestFile);
    return Status;
  }
  TerminalPrintf(L"HealthCheck: OK - test file deleted\n");

  return EFI_SUCCESS;
}

/* save the whole text buffer to file by streaming lines (safe, no big buffer) */
/* This version includes HealthCheckLogsDir *before* creating the actual log file. */
STATIC
EFI_STATUS
SaveConsoleBufferToMyLogs(VOID)
{
  EFI_STATUS Status;

  TerminalPrintf(L"SaveLogs: invoked. Beginning secure save sequence...\n");
  TerminalPrintf(L"SaveLogs: [ VER 5 ] \n");

  /* 1) Ensure directory exists and is accessible */
  EFI_FILE_PROTOCOL *LogsDir = NULL;
  TerminalPrintf(L"Step 1: Ensure directory %s exists (attempting)...\n", MYLOGS_REL_PATH);
  Status = EnsureMyLogsDir(&LogsDir);
  if (EFI_ERROR(Status) || LogsDir == NULL) {
    TerminalPrintf(L"Step 1: FAIL - cannot open or create %s: %r\n", MYLOGS_REL_PATH, Status);
    return Status;
  }
  TerminalPrintf(L"Step 1: OK - directory handle obtained. CloseRequired=%d\n", (LogsDir != gRoot) ? 1 : 0);
  BOOLEAN CloseLogsDir = (LogsDir != gRoot);

  /* 2) Quick RAM test */
  TerminalPrintf(L"Step 2: RAM quick test (%u bytes)...\n", (UINT32)RAM_TEST_BYTES);
  Status = RamQuickTest(RAM_TEST_BYTES);
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"Step 2: FAIL - RAM quick test failed: %r\n", Status);
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return Status;
  }
  TerminalPrintf(L"Step 2: OK - RAM quick test passed.\n");

  /* 3) Quick sanity: ensure there is something to save */
  TerminalPrintf(L"Step 3: Preparing to stream log content from text buffer (count=%u)...\n", (UINT32)TextBufferCount);
  if (TextBuffer == NULL || TextBufferCount == 0) {
    TerminalPrintf(L"Step 3: NOTE - buffer empty, nothing to save.\n");
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return EFI_SUCCESS;
  }

  /* 3.5) Health check: ensure creating/writing/reading/deleting small file inside LogsDir works */
  TerminalPrintf(L"Step 3.5: Performing health check on %s ...\n", MYLOGS_REL_PATH);
  CHAR16 testname[64];
  Status = HealthCheckLogsDir(LogsDir, testname, ARRAY_SIZE(testname));
  if (EFI_ERROR(Status)) {
    TerminalPrintf(L"Step 3.5: FAIL - health check failed for %s: %r\n", MYLOGS_REL_PATH, Status);
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return Status;
  }
  TerminalPrintf(L"Step 3.5: OK - health check passed (temp file used: %s)\n", testname);

  /* 4) Create unique file name and open file (we stream into it) */
  TerminalPrintf(L"Step 4: Creating unique file in %s ...\n", MYLOGS_REL_PATH);
  CHAR16 FinalName[LOG_FILENAME_MAX];
  EFI_FILE_PROTOCOL *LogFile = NULL;
  Status = CreateUniqueLogFile(LogsDir, &LogFile, FinalName, ARRAY_SIZE(FinalName));
  if (EFI_ERROR(Status) || LogFile == NULL) {
    TerminalPrintf(L"Step 4: FAIL - cannot create unique file in %s: %r\n", MYLOGS_REL_PATH, Status);
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return Status;
  }
  TerminalPrintf(L"Step 4: OK - created file '%s'\n", FinalName);

  /* 5) Write BOM (UTF-16 LE) */
  TerminalPrintf(L"Step 5: Writing BOM to file '%s' ...\n", FinalName);
  UINT8 Bom[2] = { 0xFF, 0xFE };
  UINTN WriteSize = sizeof(Bom);
  Status = LogFile->Write(LogFile, &WriteSize, Bom);
  if (EFI_ERROR(Status) || WriteSize != sizeof(Bom)) {
    TerminalPrintf(L"Step 5: FAIL - cannot write BOM to file %s: %r\n", FinalName, Status);
    if (LogFile) { EFI_STATUS d = LogFile->Delete(LogFile); if (EFI_ERROR(d)) LogFile->Close(LogFile); }
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return EFI_DEVICE_ERROR;
  }
  TerminalPrintf(L"Step 5: OK - BOM written (%u bytes)\n", (UINT32)WriteSize);

  /* 6) Stream each buffer line to the file (UTF-16LE) */
  TerminalPrintf(L"Step 6: Streaming %u lines to file '%s' ...\n", (UINT32)TextBufferCount, FinalName);

  CONST CHAR16 crlf[2] = { L'\r', L'\n' };
  UINT64 totalBytesWritten = 0;
  for (UINTN i = 0; i < TextBufferCount; ++i) {
    UINTN idx = (TextBufferHead + i) % TextBufferLines;
    CONST CHAR16 *seg = TextBuffer[idx];
    UINTN segLen = (seg != NULL) ? StrLen(seg) : 0;
    UINTN bytesToWrite = segLen * sizeof(CHAR16);

    if (bytesToWrite > 0) {
      UINTN bw = bytesToWrite;
      Status = LogFile->Write(LogFile, &bw, (VOID*)seg);
      if (EFI_ERROR(Status) || bw != bytesToWrite) {
        TerminalPrintf(L"Step 6: FAIL - write failed for line %u: %r (wrote %u of %u)\n", (UINT32)(i+1), Status, (UINT32)bw, (UINT32)bytesToWrite);
        if (LogFile) { EFI_STATUS d = LogFile->Delete(LogFile); if (EFI_ERROR(d)) LogFile->Close(LogFile); }
        if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
        return EFI_DEVICE_ERROR;
      }
      totalBytesWritten += bw;
      TerminalPrintf(L"Step 6: wrote line %u: %u chars (%u bytes)\n", (UINT32)(i+1), (UINT32)segLen, (UINT32)bw);
    } else {
      TerminalPrintf(L"Step 6: line %u is empty, skipping content write\n", (UINT32)(i+1));
    }

    /* write CRLF after every line */
    {
      UINTN bw = sizeof(crlf);
      Status = LogFile->Write(LogFile, &bw, (VOID*)crlf);
      if (EFI_ERROR(Status) || bw != sizeof(crlf)) {
        TerminalPrintf(L"Step 6: FAIL - CRLF write failed after line %u: %r\n", (UINT32)(i+1), Status);
        if (LogFile) { EFI_STATUS d = LogFile->Delete(LogFile); if (EFI_ERROR(d)) LogFile->Close(LogFile); }
        if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
        return EFI_DEVICE_ERROR;
      }
      totalBytesWritten += bw;
    }
  }

  TerminalPrintf(L"Step 6: OK - streaming complete. total bytes written (excluding BOM) = %llu\n", totalBytesWritten);

  /* 7) Close file */
  TerminalPrintf(L"Step 7: Closing file '%s' ...\n", FinalName);
  LogFile->Close(LogFile);
  TerminalPrintf(L"Step 7: OK - file closed.\n");

  TerminalPrintf(L"Step 8: Finalizing. Saved console buffer to %s\\%s\n", MYLOGS_REL_PATH, FinalName);

  if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
  return EFI_SUCCESS;
}

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

// fixed log stuff

/* Return the saved original OutputString for the provided protocol instance (This),
   or NULL if not found. */
STATIC
EFI_TEXT_STRING
GetSavedOriginalOutputString(IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *This)
{
  if (This == NULL) return NULL;

  for (UINTN i = 0; i < gPatchedCount; ++i) {
    if (gPatchedTextOuts[i].Proto == This) {
      return gPatchedTextOuts[i].OriginalOutputString;
    }
  }

  /* fallback: if we saved a global original for gST->ConOut, return it */
  if (gOriginalOutputString) return gOriginalOutputString;

  return NULL;
}


#if 0
STATIC
EFI_STATUS
EFIAPI
MyOutputString(
  IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *This,
  IN CHAR16                          *String
  )
{
  EFI_STATUS Status = EFI_SUCCESS;

  /* 1) capture into memory-backed ring if active */
  if (gCaptureMemActive && String != NULL && gCaptureMem != NULL && gCaptureMaxLines > 0 && gCaptureSlotChars > 0) {
    /* duplicate caller buffer & split safely */
    CHAR16 *Copy = DupStr(String);
    if (Copy != NULL) {
      CHAR16 *p = Copy;
      CHAR16 *seg = p;

      while (*p) {
        if (*p == L'\r' || *p == L'\n') {
          *p = L'\0';
          if (seg && *seg) {
            /* write seg into current slot (truncate if necessary) */
            UINTN slot = gCaptureHead;
            CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
            UINTN len = StrLen(seg);
            if (len > gCaptureMaxChars) len = gCaptureMaxChars;
            /* Copy exactly len CHAR16s and terminate */
            CopyMem(dst, seg, len * sizeof(CHAR16));
            dst[len] = L'\0';
            /* advance head/count */
            gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
            if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
          }
          p++;
          while (*p == L'\r' || *p == L'\n') p++;
          seg = p;
        } else {
          p++;
        }
      }

      /* final segment */
      if (seg && *seg) {
        UINTN slot = gCaptureHead;
        CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
        UINTN len = StrLen(seg);
        if (len > gCaptureMaxChars) len = gCaptureMaxChars;
        CopyMem(dst, seg, len * sizeof(CHAR16));
        dst[len] = L'\0';
        gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
        if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
      }

      FreeStr(Copy);
    }
  } else {
    /* If memory capture not active, keep previous behaviour: add to TextBuffer */
    if (String != NULL) {
      /* split and store into TextBuffer (same as TerminalPrintf splitting does) */
      CHAR16 *Copy = DupStr(String);
      if (Copy) {
        CHAR16 *p = Copy;
        CHAR16 *seg = p;
        while (*p) {
          if (*p == L'\r' || *p == L'\n') {
            *p = L'\0';
            if (seg && *seg) TerminalAddLine(seg);
            p++;
            while (*p == L'\r' || *p == L'\n') p++;
            seg = p;
          } else p++;
        }
        if (seg && *seg) TerminalAddLine(seg);
        FreeStr(Copy);
      }
    }
  }

  /* 2) forward to original output for real display (preserve behaviour) */
  if (gOriginalOutputString) {
    Status = gOriginalOutputString(This, String);
  } else {
    Status = EFI_UNSUPPORTED;
  }

  return Status;
}
#endif 

#ifdef __FIRST_WORKING_VERSION__
STATIC
EFI_STATUS
EFIAPI
MyOutputString(
  IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *This,
  IN CHAR16                          *String
  )
{
  EFI_STATUS Status = EFI_SUCCESS;

  /* 0) quick sanity */
  if (String == NULL) {
    /* just forward to original */
    EFI_TEXT_STRING orig = GetSavedOriginalOutputString(This);
    if (orig) return orig(This, String);
    return EFI_UNSUPPORTED;
  }

  /* 1) Copy the caller string so we can safely split CR/LF without touching caller memory. */
  CHAR16 *Copy = DupStr(String);
  if (Copy == NULL) {
    /* Memory allocation failed -- do not try to call TerminalPrintf here (recursion risk).
       Just forward to original (best-effort). */
    EFI_TEXT_STRING orig = GetSavedOriginalOutputString(This);
    if (orig) return orig(This, String);
    return EFI_OUT_OF_RESOURCES;
  }

  /* 2) Split into lines and store into capture area (memory-backed ring) or TextBuffer */
  CHAR16 *p = Copy;
  CHAR16 *seg = p;
  while (*p) {
    if (*p == L'\r' || *p == L'\n') {
      *p = L'\0';
      if (seg && *seg) {
        if (gCaptureMemActive && gCaptureMem && gCaptureMaxLines > 0 && gCaptureSlotChars > 0) {
          /* write into memory ring slot (truncate if necessary) */
          UINTN slot = gCaptureHead;
          CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
          UINTN len = StrLen(seg);
          if (len > gCaptureMaxChars) len = gCaptureMaxChars;
          CopyMem(dst, seg, len * sizeof(CHAR16));
          dst[len] = L'\0';
          gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
          if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
        } else {
          /* fallback to TextBuffer (this does allocations but is your existing ring) */
          TerminalAddLine(seg);
        }
      }
      p++;
      while (*p == L'\r' || *p == L'\n') p++;
      seg = p;
    } else {
      p++;
    }
  }

  /* last segment (if any) */
  if (seg && *seg) {
    if (gCaptureMemActive && gCaptureMem && gCaptureMaxLines > 0 && gCaptureSlotChars > 0) {
      UINTN slot = gCaptureHead;
      CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
      UINTN len = StrLen(seg);
      if (len > gCaptureMaxChars) len = gCaptureMaxChars;
      CopyMem(dst, seg, len * sizeof(CHAR16));
      dst[len] = L'\0';
      gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
      if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
    } else {
      TerminalAddLine(seg);
    }
  }

  FreeStr(Copy);

  /* 3) Forward to the original OutputString for this protocol instance */
  EFI_TEXT_STRING orig = GetSavedOriginalOutputString(This);
  if (orig) {
    Status = orig(This, String);
  } else {
    Status = EFI_UNSUPPORTED;
  }

  return Status;
}
#else
/* helper: find patched entry index for a protocol instance; returns -1 if not found */
STATIC
INTN
FindPatchedIndex(IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *This)
{
  if (This == NULL) return -1;
  for (UINTN i = 0; i < gPatchedCount; ++i) {
    if (gPatchedTextOuts[i].Proto == This) return (INTN)i;
  }
  return -1;
}

#ifdef __FIRST_WORKING_VERSION__
/* helper: compare with last committed captured line (returns TRUE if equal) */
STATIC
BOOLEAN
IsDuplicateLastCaptured(CONST CHAR16 *Line)
{
  if (Line == NULL || StrLen(Line) == 0) return FALSE;
  /* If memory capture active, check last slot; otherwise check last TextBuffer entry */
  if (gCaptureMem && gCaptureCount > 0) {
    UINTN lastIdx;
    if (gCaptureCount < gCaptureMaxLines) {
      lastIdx = (gCaptureHead + gCaptureMaxLines + gCaptureCount - 1) % gCaptureMaxLines;
    } else {
      lastIdx = (gCaptureHead + gCaptureMaxLines - 1) % gCaptureMaxLines;
    }
    CHAR16 *last = gCaptureMem + lastIdx * gCaptureSlotChars;
    if (last && StrCmp(last, Line) == 0) return TRUE;
    return FALSE;
  } else if (TextBufferCount > 0) {
    UINTN idx = (TextBufferHead + TextBufferCount - 1) % TextBufferLines;
    if (TextBuffer[idx] && StrCmp(TextBuffer[idx], Line) == 0) return TRUE;
  }
  return FALSE;
}
#else
/* Return TRUE if Line equals the last committed captured line. */
STATIC
BOOLEAN
IsDuplicateLastCaptured(CONST CHAR16 *Line)
{
  if (Line == NULL || StrLen(Line) == 0) return FALSE;
  if (gCaptureMem == NULL || gCaptureCount == 0) return FALSE;

  /* last index is always (head + max - 1) % max (head points to next write slot). */
  UINTN lastIdx = (gCaptureHead + gCaptureMaxLines + gCaptureMaxLines - 1) % gCaptureMaxLines;
  /* simplify: (gCaptureHead + gCaptureMaxLines - 1) % gCaptureMaxLines */
  lastIdx = (gCaptureHead + gCaptureMaxLines - 1) % gCaptureMaxLines;

  CHAR16 *last = gCaptureMem + lastIdx * gCaptureSlotChars;
  if (last == NULL) return FALSE;
  return (StrCmp(last, Line) == 0);
}

#endif

STATIC
EFI_STATUS
EFIAPI
MyOutputString(
  IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *This,
  IN CHAR16                          *String
  )
{
  EFI_STATUS Status = EFI_SUCCESS;

  /* forward to original at end; look up original pointer for this instance */
  EFI_TEXT_STRING orig = NULL;
  INTN pidx = FindPatchedIndex(This);
  if (pidx >= 0) orig = gPatchedTextOuts[pidx].OriginalOutputString;
  else orig = gOriginalOutputString;

  /* If no string, just forward */
  if (String == NULL) {
    if (orig) return orig(This, String);
    return EFI_UNSUPPORTED;
  }

  /* If we have a patched entry with accumulator, append input and commit complete lines only */
  if (pidx >= 0 && gPatchedTextOuts[pidx].AccumBuf) {
    PATCHED_TEXTOUT_ENTRY *E = &gPatchedTextOuts[pidx];
    CHAR16 *acc = E->AccumBuf;
    UINTN accCap = E->AccumBufChars;
    UINTN accLen = StrLen(acc);

    /* make a local copy to iterate the incoming string safely */
    CHAR16 *in = DupStr(String);
    if (in == NULL) {
      /* allocation failed — best-effort: forward and return orig result */
      if (orig) return orig(This, String);
      return EFI_OUT_OF_RESOURCES;
    }

    CHAR16 *q = in;
    while (*q) {
      /* append character-by-character until we hit CR/LF */
      if (*q == L'\r' || *q == L'\n') {
        /* commit accumulator as a complete line if non-empty */
        if (accLen > 0) {
          acc[accLen] = L'\0';
          /* dedupe immediate duplicates */
          if (!IsDuplicateLastCaptured(acc)) {
            if (gCaptureMemActive && gCaptureMem) {
              /* write into ring */
              UINTN slot = gCaptureHead;
              CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
              UINTN copyLen = (accLen > gCaptureMaxChars) ? gCaptureMaxChars : accLen;
              CopyMem(dst, acc, copyLen * sizeof(CHAR16));
              dst[copyLen] = L'\0';
              gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
              if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
            } else {
              TerminalAddLine(acc);
            }
          }
          /* reset accumulator */
          acc[0] = L'\0';
          accLen = 0;
        }
        /* skip contiguous CR/LF */
        q++;
        while (*q == L'\r' || *q == L'\n') q++;
        continue;
      }

      /* normal character - append if space allows */
      if (accLen + 1 < accCap) {
        acc[accLen++] = *q;
        acc[accLen] = L'\0';
      } else {
        /* accumulator full: truncate, commit immediately */
        acc[accCap - 1] = L'\0';
        if (!IsDuplicateLastCaptured(acc)) {
          if (gCaptureMemActive && gCaptureMem) {
            UINTN slot = gCaptureHead;
            CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
            UINTN copyLen = (accCap - 1 > gCaptureMaxChars) ? gCaptureMaxChars : (accCap - 1);
            CopyMem(dst, acc, copyLen * sizeof(CHAR16));
            dst[copyLen] = L'\0';
            gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
            if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
          } else {
            TerminalAddLine(acc);
          }
        }
        /* clear accumulator to continue */
        acc[0] = L'\0';
        accLen = 0;
      }
      q++;
    }

    FreeStr(in);
  } else {
    /* No accumulator available - fallback to previous behaviour (split and store immediately) */
    CHAR16 *Copy = DupStr(String);
    if (Copy) {
      CHAR16 *p = Copy;
      CHAR16 *seg = p;
      while (*p) {
        if (*p == L'\r' || *p == L'\n') {
          *p = L'\0';
          if (seg && *seg) {
            if (!IsDuplicateLastCaptured(seg)) {
              if (gCaptureMemActive && gCaptureMem) {
                UINTN slot = gCaptureHead;
                CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
                UINTN len = StrLen(seg);
                if (len > gCaptureMaxChars) len = gCaptureMaxChars;
                CopyMem(dst, seg, len * sizeof(CHAR16));
                dst[len] = L'\0';
                gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
                if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
              } else {
                TerminalAddLine(seg);
              }
            }
          }
          p++;
          while (*p == L'\r' || *p == L'\n') p++;
          seg = p;
        } else p++;
      }
      if (seg && *seg) {
        if (!IsDuplicateLastCaptured(seg)) {
          if (gCaptureMemActive && gCaptureMem) {
            UINTN slot = gCaptureHead;
            CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
            UINTN len = StrLen(seg);
            if (len > gCaptureMaxChars) len = gCaptureMaxChars;
            CopyMem(dst, seg, len * sizeof(CHAR16));
            dst[len] = L'\0';
            gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
            if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
          } else {
            TerminalAddLine(seg);
          }
        }
      }
      FreeStr(Copy);
    }
  }

  /* 3) Now forward to the original OutputString for this protocol instance */
  if (orig) {
    Status = orig(This, String);
  } else {
    Status = EFI_UNSUPPORTED;
  }

  return Status;
}
#endif

/* Start memory capture with capacity lines x charsPerLine.
   Caps are enforced to prevent runaway allocations.
   Returns EFI_SUCCESS on success.
*/
STATIC
EFI_STATUS
StartCaptureMem(UINTN lines, UINTN charsPerLine)
{
  if (lines == 0 || charsPerLine == 0) return EFI_INVALID_PARAMETER;

  /* safe caps */
  const UINTN MAX_LINES_CAP = 10000;      /* arbitrary cap */
  const UINTN MAX_CHARS_CAP = 4096;       /* per-line cap */
  const UINTN MAX_TOTAL_BYTES = 64 * 1024 * 1024; /* 64 MB total cap */

  if (lines > MAX_LINES_CAP) lines = MAX_LINES_CAP;
  if (charsPerLine > MAX_CHARS_CAP) charsPerLine = MAX_CHARS_CAP;

  UINTN slotChars = charsPerLine + 1; /* for NUL */
  UINTN totalBytes = lines * slotChars * sizeof(CHAR16);

  if (totalBytes == 0 || totalBytes > MAX_TOTAL_BYTES) {
    return EFI_OUT_OF_RESOURCES;
  }

  /* free old capture if present */
  if (gCaptureMem) {
    FreePool(gCaptureMem);
    gCaptureMem = NULL;
  }

  CHAR16 *buf = AllocateZeroPool(totalBytes);
  if (!buf) return EFI_OUT_OF_RESOURCES;

  /* initialize meta */
  gCaptureMem = buf;
  gCaptureMemBytes = totalBytes;
  gCaptureMaxLines = lines;
  gCaptureMaxChars = charsPerLine;
  gCaptureSlotChars = slotChars;
  gCaptureHead = 0;
  gCaptureCount = 0;
  gCaptureMemActive = TRUE;

  /* print status using original output so we don't re-enter capture logic unexpectedly */
  if (gOriginalOutputString) {
    gOriginalOutputString(gST->ConOut, L"capmem: memory capture started\n");
  } else {
    Print(L"capmem: memory capture started\n");
  }

  return EFI_SUCCESS;
}

STATIC
VOID
StopCaptureMem(VOID)
{
  gCaptureMemActive = FALSE;
  if (gOriginalOutputString) {
    gOriginalOutputString(gST->ConOut, L"capmem: memory capture stopped\n");
  } else {
    Print(L"capmem: memory capture stopped\n");
  }
  /* keep buffer allocated (user may want to save it). Caller may free with a 'freecap' cmd if wanted. */
}


#if 0
/* safe diagnostic print using saved original for system console (if available) */
STATIC
VOID
SafeConsolePrint(CONST CHAR16 *Fmt, ...)
{
  if (gOriginalOutputString == NULL) {
    /* fallback: Print (this may route through wrapper) */
    VA_LIST Args;
    VA_START(Args, Fmt);
    UnicodeVSPrint(gTmpLineBuffer, sizeof(gTmpLineBuffer), Fmt, Args); // allocate gTmpLineBuffer static
    VA_END(Args);
    Print(L"%s", gTmpLineBuffer);
    return;
  }

  /* format into local buffer then call saved original */
  CHAR16 Tmp[1024];
  VA_LIST Args;
  VA_START(Args, Fmt);
  UnicodeVSPrint(Tmp, sizeof(Tmp), Fmt, Args);
  VA_END(Args);

  /* gOriginalOutputString expects This pointer for the console instance */
  gOriginalOutputString(gST->ConOut, Tmp);
}
#endif

STATIC
VOID
SafeConsolePrint(CONST CHAR16 *Fmt, ...)
{
  VA_LIST Args;
  VA_START(Args, Fmt);

  /* Format into our temp buffer (pass size in bytes like the rest of file does) */
  UnicodeVSPrint(gTmpLineBuffer, sizeof(gTmpLineBuffer), Fmt, Args);

  VA_END(Args);

  /* If we have a saved original for the system console, call it directly to avoid recursion.
     Otherwise fallback to Print (may route via wrapper). */
  if (gOriginalOutputString) {
    gOriginalOutputString(gST->ConOut, gTmpLineBuffer);
  } else {
    Print(L"%s", gTmpLineBuffer);
  }
}



/* Free allocated capture memory immediately */
STATIC
VOID
FreeCaptureMem(VOID)
{
  if (gCaptureMem) {
    FreePool(gCaptureMem);
    gCaptureMem = NULL;
  }
  gCaptureMemBytes = 0;
  gCaptureMaxLines = 0;
  gCaptureMaxChars = 0;
  gCaptureSlotChars = 0;
  gCaptureHead = 0;
  gCaptureCount = 0;
  gCaptureMemActive = FALSE;
}

#ifdef __FIRST_WORKING_VERSION__
STATIC
EFI_STATUS
SaveCaptureMemToMyLogs(VOID)
{
  if (gCaptureMem == NULL || gCaptureCount == 0) {
    TerminalPrintf(L"savecap: no captured lines present\n");
    return EFI_SUCCESS;
  }

  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *LogsDir = NULL;
  Status = EnsureMyLogsDir(&LogsDir);
  if (EFI_ERROR(Status) || LogsDir == NULL) {
    TerminalPrintf(L"savecap: cannot open/create myLogs: %r\n", Status);
    return Status;
  }
  BOOLEAN CloseLogsDir = (LogsDir != gRoot);

  /* Health check optional (could call HealthCheckLogsDir) - omitted here to stay compact */

  CHAR16 FinalName[LOG_FILENAME_MAX];
  EFI_FILE_PROTOCOL *LogFile = NULL;
  Status = CreateUniqueLogFile(LogsDir, &LogFile, FinalName, ARRAY_SIZE(FinalName));
  if (EFI_ERROR(Status) || LogFile == NULL) {
    TerminalPrintf(L"savecap: CreateUniqueLogFile failed: %r\n", Status);
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return Status;
  }

  /* write BOM */
  UINT8 Bom[2] = { 0xFF, 0xFE };
  UINTN write = sizeof(Bom);
  Status = LogFile->Write(LogFile, &write, Bom);
  if (EFI_ERROR(Status) || write != sizeof(Bom)) {
    TerminalPrintf(L"savecap: cannot write BOM: %r\n", Status);
    LogFile->Delete(LogFile);
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return EFI_DEVICE_ERROR;
  }

  CONST CHAR16 crlf[2] = { L'\r', L'\n' };

  /* compute start index (oldest) */
  UINTN start;
  if (gCaptureCount < gCaptureMaxLines) {
    start = 0;
  } else {
    start = gCaptureHead; /* head points at next slot, so oldest is head */
  }

  /* iterate gCaptureCount lines in order */
  for (UINTN i = 0; i < gCaptureCount; ++i) {
    UINTN idx = (start + i) % gCaptureMaxLines;
    CHAR16 *slot = gCaptureMem + idx * gCaptureSlotChars;
    UINTN segLen = StrLen(slot);
    if (segLen > 0) {
      UINTN bytesToWrite = segLen * sizeof(CHAR16);
      UINTN bw = bytesToWrite;
      Status = LogFile->Write(LogFile, &bw, (VOID*)slot);
      if (EFI_ERROR(Status) || bw != bytesToWrite) {
        TerminalPrintf(L"savecap: write failed at line %u: %r\n", (UINT32)i, Status);
        LogFile->Delete(LogFile);
        if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
        return EFI_DEVICE_ERROR;
      }
    }
    /* write CRLF */
    UINTN bw = sizeof(crlf);
    Status = LogFile->Write(LogFile, &bw, (VOID*)crlf);
    if (EFI_ERROR(Status) || bw != sizeof(crlf)) {
      TerminalPrintf(L"savecap: CRLF write failed at line %u: %r\n", (UINT32)i, Status);
      LogFile->Delete(LogFile);
      if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
      return EFI_DEVICE_ERROR;
    }
  }

  /* close file */
  LogFile->Close(LogFile);
  if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);

  TerminalPrintf(L"savecap: saved %u lines to %s\\%s\n", (UINT32)gCaptureCount, MYLOGS_REL_PATH, FinalName);
  return EFI_SUCCESS;
}
#else
/* Commit any non-empty accumulators into the capture ring or TextBuffer.
   This is safe to call just before saving. */
STATIC
VOID
FlushAllAccumulators(VOID)
{
  for (UINTN i = 0; i < gPatchedCount; ++i) {
    PATCHED_TEXTOUT_ENTRY *E = &gPatchedTextOuts[i];
    if (E == NULL || E->AccumBuf == NULL) continue;
    UINTN len = StrLen(E->AccumBuf);
    if (len == 0) continue;

    /* dedupe */
    if (IsDuplicateLastCaptured(E->AccumBuf)) {
      /* clear accumulator */
      E->AccumBuf[0] = L'\0';
      continue;
    }

    if (gCaptureMemActive && gCaptureMem) {
      UINTN slot = gCaptureHead;
      CHAR16 *dst = gCaptureMem + slot * gCaptureSlotChars;
      UINTN copyLen = (len > gCaptureMaxChars) ? gCaptureMaxChars : len;
      CopyMem(dst, E->AccumBuf, copyLen * sizeof(CHAR16));
      dst[copyLen] = L'\0';
      gCaptureHead = (gCaptureHead + 1) % gCaptureMaxLines;
      if (gCaptureCount < gCaptureMaxLines) gCaptureCount++;
    } else {
      TerminalAddLine(E->AccumBuf);
    }

    /* clear accumulator after commit */
    E->AccumBuf[0] = L'\0';
  }
}

STATIC
EFI_STATUS
SaveCaptureMemToMyLogs(VOID)
{
  if (gCaptureMem == NULL || gCaptureCount == 0) {
    TerminalPrintf(L"savecap: no captured lines present\n");
    return EFI_SUCCESS;
  }

  EFI_STATUS Status;
  EFI_FILE_PROTOCOL *LogsDir = NULL;
  Status = EnsureMyLogsDir(&LogsDir);
  if (EFI_ERROR(Status) || LogsDir == NULL) {
    TerminalPrintf(L"savecap: cannot open/create myLogs: %r\n", Status);
    return Status;
  }
  BOOLEAN CloseLogsDir = (LogsDir != gRoot);

  CHAR16 FinalName[LOG_FILENAME_MAX];
  EFI_FILE_PROTOCOL *LogFile = NULL;
  Status = CreateUniqueLogFile(LogsDir, &LogFile, FinalName, ARRAY_SIZE(FinalName));
  if (EFI_ERROR(Status) || LogFile == NULL) {
    TerminalPrintf(L"savecap: CreateUniqueLogFile failed: %r\n", Status);
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return Status;
  }

  /* write BOM */
  UINT8 Bom[2] = { 0xFF, 0xFE };
  UINTN write = sizeof(Bom);
  Status = LogFile->Write(LogFile, &write, Bom);
  if (EFI_ERROR(Status) || write != sizeof(Bom)) {
    TerminalPrintf(L"savecap: cannot write BOM: %r\n", Status);
    LogFile->Delete(LogFile);
    if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
    return EFI_DEVICE_ERROR;
  }

  CONST CHAR16 crlf[2] = { L'\r', L'\n' };

  /* === SAFETY: suspend capture, flush accumulators, snapshot indexes === */
  BOOLEAN oldActive = gCaptureMemActive;
  gCaptureMemActive = FALSE;      /* stop new writes into ring while we snapshot */
  FlushAllAccumulators();         /* commit partial lines into ring */

  /* snapshot values (local copies) */
  UINTN snapshotCount = gCaptureCount;
  UINTN snapshotHead = gCaptureHead;
  UINTN snapshotMax = gCaptureMaxLines;
  UINTN snapshotSlotChars = gCaptureSlotChars;
  CHAR16 *snapshotMem = gCaptureMem;

  /* compute start index (oldest) */
  UINTN start;
  if (snapshotCount == 0) {
    start = 0;
  } else if (snapshotCount < snapshotMax) {
    start = 0;
  } else {
    start = snapshotHead; /* head points at next slot, so oldest is head */
  }

  /* iterate snapshotCount lines in order */
  for (UINTN i = 0; i < snapshotCount; ++i) {
    UINTN idx = (start + i) % snapshotMax;
    CHAR16 *slot = snapshotMem + idx * snapshotSlotChars;
    UINTN segLen = StrLen(slot);
    if (segLen > 0) {
      UINTN bytesToWrite = segLen * sizeof(CHAR16);
      UINTN bw = bytesToWrite;
      Status = LogFile->Write(LogFile, &bw, (VOID*)slot);
      if (EFI_ERROR(Status) || bw != bytesToWrite) {
        TerminalPrintf(L"savecap: write failed at line %u: %r\n", (UINT32)i, Status);
        LogFile->Delete(LogFile);
        if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
        /* restore capture state before returning */
        gCaptureMemActive = oldActive;
        return EFI_DEVICE_ERROR;
      }
    }
    /* write CRLF */
    UINTN bw = sizeof(crlf);
    Status = LogFile->Write(LogFile, &bw, (VOID*)crlf);
    if (EFI_ERROR(Status) || bw != sizeof(crlf)) {
      TerminalPrintf(L"savecap: CRLF write failed at line %u: %r\n", (UINT32)i, Status);
      LogFile->Delete(LogFile);
      if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);
      gCaptureMemActive = oldActive;
      return EFI_DEVICE_ERROR;
    }
  }

  /* close file */
  LogFile->Close(LogFile);
  if (CloseLogsDir && LogsDir) LogsDir->Close(LogsDir);

  /* restore capture activity */
  gCaptureMemActive = oldActive;

  TerminalPrintf(L"savecap: saved %u lines to %s\\%s\n", (UINT32)snapshotCount, MYLOGS_REL_PATH, FinalName);
  return EFI_SUCCESS;
}
#endif

// version 1
#if 0
STATIC VOID DoCmdCapMem(CONST CHAR16 *Arg)
{
  if (Arg == NULL) {
    TerminalPrintf(L"capmem: usage: capmem start <lines> <chars-per-line> | stop | status | save | free\n");
    return;
  }

  /* copy arg to lowercase token space */
  CHAR16 local[LINE_BUFFER_SIZE];
  StrCpyS(local, ARRAY_SIZE(local), Arg);
  for (CHAR16 *p = local; *p; ++p) if (*p >= L'A' && *p <= L'Z') *p = *p - L'A' + L'a';

  CHAR16 *tok = StrStr(local, L" ");
  if (!tok) {
    /* single-word commands */
    if (StrCmp(local, L"stop") == 0) {
      StopCaptureMem();
      return;
    } else if (StrCmp(local, L"status") == 0) {
      TerminalPrintf(L"capmem: active=%d lines=%u/max=%u chars=%u\n",
                     gCaptureMemActive ? 1 : 0, (UINT32)gCaptureCount, (UINT32)gCaptureMaxLines, (UINT32)gCaptureMaxChars);
      return;
    } else if (StrCmp(local, L"save") == 0) {
      SaveCaptureMemToMyLogs();
      return;
    } else if (StrCmp(local, L"free") == 0) {
      FreeCaptureMem();
      TerminalPrintf(L"capmem: freed capture memory\n");
      return;
    }
  } else {
    /* token is first word, remainder in tok+1 */
    *tok = L'\0'; tok++;
    while (*tok == L' ') tok++;
    if (StrCmp(local, L"start") == 0) {
      /* parse two numbers: lines and chars */
      UINTN lines = StrToUintn(tok);
      /* find second number */
      CHAR16 *space = StrStr(tok, L" ");
      UINTN chars = 256;
      if (space) {
        chars = StrToUintn(space + 1);
      }
      if (lines == 0) lines = 2000;   /* default if not provided */
      if (chars == 0) chars = 512;
      EFI_STATUS st = StartCaptureMem(lines, chars);
      if (EFI_ERROR(st)) TerminalPrintf(L"capmem: cannot start: %r\n", st);
      return;
    }
  }

  TerminalPrintf(L"capmem: unknown or malformed command\n");
}
#endif

STATIC VOID DoCmdCapMem(CONST CHAR16 *Arg)
{
	/*
  if (Arg == NULL) {
    TerminalPrintf(L"capmem: usage:\n");
    TerminalPrintf(L"  capmem start <lines> <chars-per-line>  - allocate and start capture\n");
    TerminalPrintf(L"  capmem stop                            - stop capture and unpatch protocols\n");
    TerminalPrintf(L"  capmem status                          - show status\n");
    TerminalPrintf(L"  capmem save                            - save captured buffer to \\EFI\\Boot\\myLogs\\\n");
    TerminalPrintf(L"  capmem free                            - free capture buffer (must be stopped first)\n");
    return;
  }
  */

  /* Make a lowercase copy to parse the first token */
  CHAR16 cmd[LINE_BUFFER_SIZE];
  StrCpyS(cmd, ARRAY_SIZE(cmd), Arg);
  for (CHAR16 *p = cmd; *p; ++p) if (*p >= L'A' && *p <= L'Z') *p = *p - L'A' + L'a';

  /* Find the first token (word) */
  CHAR16 *rest = StrStr(cmd, L" ");
  if (rest) {
    *rest = L'\0';
    rest++;
    while (*rest == L' ') rest++;
  }

  if (StrCmp(cmd, L"start") == 0) {
    /* parse numbers from original Arg (not lowercase copy) to preserve digits */
    /* Arg points to original string passed from main loop (after 'capmem ') */
    /* expected format: "start <lines> <chars>" */
    /* Extract numbers: reuse StrToUintn which reads digits at start of string */
    CHAR16 *orig = (CHAR16*)Arg;
    /* skip 'start' */
    CHAR16 *tok = StrStr(orig, L" ");
    if (!tok) {
      TerminalPrintf(L"capmem: start requires at least number of lines\n");
      return;
    }
    tok++;
    while (*tok == L' ') tok++;
    UINTN lines = StrToUintn(tok);
    /* find next number */
    CHAR16 *tok2 = StrStr(tok, L" ");
    UINTN chars = 0;
    if (tok2) {
      tok2++;
      while (*tok2 == L' ') tok2++;
      chars = StrToUintn(tok2);
    }

    if (lines == 0) lines = 2000;   /* reasonable default */
    if (chars == 0) chars = 512;

    TerminalPrintf(L"capmem: allocating %u lines x %u chars/line ...\n", (UINT32)lines, (UINT32)chars);
    EFI_STATUS st = StartCaptureMem(lines, chars);
    if (EFI_ERROR(st)) {
      TerminalPrintf(L"capmem: StartCaptureMem failed: %r\n", st);
      return;
    }

    /* Now enable protocol-wide patching so every SimpleTextOut is hooked. */
    st = EnableAllTextOutPatch();
    if (EFI_ERROR(st)) {
      TerminalPrintf(L"capmem: EnableAllTextOutPatch failed: %r\n", st);
      /* free buffer to avoid leaked memory if patch fails */
      FreeCaptureMem();
      return;
    }

    /* mark global state if desired */
    gConsoleCaptureEnabled = TRUE;
    TerminalPrintf(L"capmem: capture started and hooks installed\n");
    return;
  }

  if (StrCmp(cmd, L"stop") == 0) {
    /* stop writing to capture area, then unpatch protocols */
    if (!gCaptureMemActive && !gConsoleCaptureEnabled) {
      TerminalPrintf(L"capmem: not currently capturing\n");
      return;
    }

    /* first stop writes so wrapper stops touching buffer */
    StopCaptureMem();

    /* then unpatch all protocol instances */
    DisableAllTextOutPatch();

    gConsoleCaptureEnabled = FALSE;
    TerminalPrintf(L"capmem: capture stopped and hooks removed\n");
    return;
  }

  if (StrCmp(cmd, L"status") == 0) {
    TerminalPrintf(L"capmem: active=%d buffer_allocated=%d lines_stored=%u capacity=%u chars_per_line=%u\n",
                   gCaptureMemActive ? 1 : 0,
                   (gCaptureMem != NULL) ? 1 : 0,
                   (UINT32)gCaptureCount,
                   (UINT32)gCaptureMaxLines,
                   (UINT32)gCaptureMaxChars);
    return;
  }

  if (StrCmp(cmd, L"save") == 0) {
    EFI_STATUS st = SaveCaptureMemToMyLogs();
    if (EFI_ERROR(st)) {
      TerminalPrintf(L"capmem: SaveCaptureMemToMyLogs failed: %r\n", st);
    }
    return;
  }

  if (StrCmp(cmd, L"free") == 0) {
    if (gConsoleCaptureEnabled) {
      TerminalPrintf(L"capmem: must stop capture before freeing memory (run 'capmem stop')\n");
      return;
    }
    FreeCaptureMem();
    TerminalPrintf(L"capmem: capture memory freed\n");
    return;
  }

  TerminalPrintf(L"capmem: unknown subcommand '%s'\n", cmd);
}

#if 0
/* Patch a single protocol instance if not already patched */
STATIC
EFI_STATUS
PatchOneTextOut(IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *Proto)
{
  if (Proto == NULL) return EFI_INVALID_PARAMETER;

  /* check if already patched */
  for (UINTN i = 0; i < gPatchedCount; ++i) {
    if (gPatchedTextOuts[i].Proto == Proto) return EFI_SUCCESS;
  }

  if (gPatchedCount >= MAX_PATCHED_TEXTOUTS) {
    return EFI_OUT_OF_RESOURCES;
  }

  /* save original and patch */
  gPatchedTextOuts[gPatchedCount].Proto = Proto;
  gPatchedTextOuts[gPatchedCount].OriginalOutputString = (EFI_TEXT_STRING)Proto->OutputString;
  Proto->OutputString = (EFI_TEXT_STRING)MyOutputString;
  gPatchedCount++;

  return EFI_SUCCESS;
}
#endif


#ifdef __FIRST_WORKING_VERSION__
STATIC
EFI_STATUS
PatchOneTextOut(IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *Proto)
{
  if (Proto == NULL) return EFI_INVALID_PARAMETER;

  /* avoid duplicates */
  for (UINTN i = 0; i < gPatchedCount; ++i) {
    if (gPatchedTextOuts[i].Proto == Proto) return EFI_SUCCESS;
  }

  if (gPatchedCount >= MAX_PATCHED_TEXTOUTS) return EFI_OUT_OF_RESOURCES;

  gPatchedTextOuts[gPatchedCount].Proto = Proto;
  gPatchedTextOuts[gPatchedCount].OriginalOutputString = (EFI_TEXT_STRING)Proto->OutputString;

  /* If this is the system console pointer, store a global fallback */
  if (Proto == gST->ConOut) {
    gOriginalOutputString = (EFI_TEXT_STRING)Proto->OutputString;
  }

  /* install wrapper */
  Proto->OutputString = (EFI_TEXT_STRING)MyOutputString;

  gPatchedCount++;
  return EFI_SUCCESS;
}
#else
STATIC
EFI_STATUS
PatchOneTextOut(IN EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *Proto)
{
  if (Proto == NULL) return EFI_INVALID_PARAMETER;

  /* already patched? */
  for (UINTN i = 0; i < gPatchedCount; ++i) {
    if (gPatchedTextOuts[i].Proto == Proto) return EFI_SUCCESS;
  }

  if (gPatchedCount >= MAX_PATCHED_TEXTOUTS) return EFI_OUT_OF_RESOURCES;

  PATCHED_TEXTOUT_ENTRY *E = &gPatchedTextOuts[gPatchedCount];
  ZeroMem(E, sizeof(*E));
  E->Proto = Proto;
  E->OriginalOutputString = (EFI_TEXT_STRING)Proto->OutputString;
  E->AccumBufChars = ACCUM_CHARS_DEFAULT;
  E->AccumBuf = AllocateZeroPool(E->AccumBufChars * sizeof(CHAR16));
  if (E->AccumBuf == NULL) {
    /* out of memory - don't patch this instance */
    E->Proto = NULL;
    E->OriginalOutputString = NULL;
    return EFI_OUT_OF_RESOURCES;
  }

  /* save global fallback for system console if encountered */
  if (Proto == gST->ConOut) {
    gOriginalOutputString = E->OriginalOutputString;
  }

  /* install wrapper */
  Proto->OutputString = (EFI_TEXT_STRING)MyOutputString;

  gPatchedCount++;
  return EFI_SUCCESS;
}

#endif


#ifdef __FIRST_WORKING_VERSION__
/* Restore a single entry */
STATIC
VOID
UnpatchAllSimpleTextOuts(VOID)
{
  for (UINTN i = 0; i < gPatchedCount; ++i) {
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *Proto = gPatchedTextOuts[i].Proto;
    if (Proto) {
      /* only restore if our wrapper is installed (avoid stomping third-party changes) */
      if (Proto->OutputString == (EFI_TEXT_STRING)MyOutputString) {
        Proto->OutputString = gPatchedTextOuts[i].OriginalOutputString;
      }
    }
    gPatchedTextOuts[i].Proto = NULL;
    gPatchedTextOuts[i].OriginalOutputString = NULL;
  }
  gPatchedCount = 0;
}
#else
STATIC
VOID
UnpatchAllSimpleTextOuts(VOID)
{
  for (UINTN i = 0; i < gPatchedCount; ++i) {
    PATCHED_TEXTOUT_ENTRY *E = &gPatchedTextOuts[i];
    if (E->Proto) {
      if (E->Proto->OutputString == (EFI_TEXT_STRING)MyOutputString) {
        E->Proto->OutputString = E->OriginalOutputString;
      }
      E->Proto = NULL;
    }
    if (E->AccumBuf) { FreePool(E->AccumBuf); E->AccumBuf = NULL; }
    E->OriginalOutputString = NULL;
    E->AccumBufChars = 0;
  }
  gPatchedCount = 0;
}

#endif

/* Find all handles that support SimpleTextOut and patch them */
STATIC
EFI_STATUS
PatchAllSimpleTextOuts(VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *HandleBuffer = NULL;
  UINTN HandleCount = 0;

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiSimpleTextOutProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status)) {
    return Status;
  }

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *Proto = NULL;
    Status = gBS->HandleProtocol(HandleBuffer[i], &gEfiSimpleTextOutProtocolGuid, (VOID**)&Proto);
    if (EFI_ERROR(Status) || Proto == NULL) continue;
    /* Attempt to patch */
    PatchOneTextOut(Proto);
  }

  if (HandleBuffer) {
    FreePool(HandleBuffer);
  }

  return EFI_SUCCESS;
}

/* This runs at TPL_CALLBACK (RegisterProtocolNotify uses default TPL). */
STATIC
VOID
EFIAPI
SimpleTextOutNotifyCallback(IN EFI_EVENT Event, IN VOID *Context)
{
  /* event signalled -> locate newly installed protocol instances and patch them */
  PatchAllSimpleTextOuts();
  /* Note: no CloseEvent here; event stays registered until we explicitly disable it. */
}

STATIC
EFI_STATUS
EnableAllTextOutPatch(VOID)
{
  EFI_STATUS Status;

  /* First patch existing instances */
  Status = PatchAllSimpleTextOuts();
  if (EFI_ERROR(Status)) {
    // continuing can still try to register notify; but report
    TerminalPrintf(L"patch: PatchAllSimpleTextOuts failed: %r\n", Status);
  }

  /* Create notify event to catch future installs */
  if (gSimpleTextOutNotifyEvent == NULL) {
    Status = gBS->CreateEvent(EVT_NOTIFY_SIGNAL, TPL_CALLBACK, SimpleTextOutNotifyCallback, NULL, &gSimpleTextOutNotifyEvent);
    if (EFI_ERROR(Status)) {
      TerminalPrintf(L"patch: CreateEvent failed: %r\n", Status);
      return Status;
    }

    Status = gBS->RegisterProtocolNotify(&gEfiSimpleTextOutProtocolGuid, gSimpleTextOutNotifyEvent, &gSimpleTextOutNotifyReg);
    if (EFI_ERROR(Status)) {
      gBS->CloseEvent(gSimpleTextOutNotifyEvent);
      gSimpleTextOutNotifyEvent = NULL;
      TerminalPrintf(L"patch: RegisterProtocolNotify failed: %r\n", Status);
      return Status;
    }
  }

  return EFI_SUCCESS;
}

STATIC
VOID
DisableAllTextOutPatch(VOID)
{
  /* restore original pointers */
  UnpatchAllSimpleTextOuts();

  /* close/clear notification event */
  if (gSimpleTextOutNotifyEvent) {
    gBS->CloseEvent(gSimpleTextOutNotifyEvent);
    gSimpleTextOutNotifyEvent = NULL;
    gSimpleTextOutNotifyReg = NULL;
  }
}

STATIC
VOID
EnableConsoleCapture(VOID)
{
  if (gConsoleCaptureEnabled) {
    TerminalPrintf(L"capture: already enabled\n");
    return;
  }

  /* patch ALL SimpleTextOut instances and register notify for future ones */
  EFI_STATUS st = EnableAllTextOutPatch();
  if (EFI_ERROR(st)) {
    TerminalPrintf(L"capture: failed to enable protocol-wide patch: %r\n", st);
    return;
  }

  gConsoleCaptureEnabled = TRUE;
  TerminalPrintf(L"capture: enabled - console output will be recorded from all TextOut instances\n");
}

STATIC
VOID
DisableConsoleCapture(VOID)
{
  if (!gConsoleCaptureEnabled) {
    TerminalPrintf(L"capture: already disabled\n");
    return;
  }

  /* restore originals and remove notify */
  DisableAllTextOutPatch();

  gConsoleCaptureEnabled = FALSE;
  TerminalPrintf(L"capture: disabled\n");
}

/*
  DiagnosticCheckCaptureBuffers()

  Non-invasive debug helper that inspects:
   - memory-backed capture (gCaptureMem / ring state)
   - textual ring buffer (TextBuffer)
   - patched SimpleTextOut entries (gPatchedTextOuts)
  It prints statistics and small samples so you can see:
   - consecutive duplicate streaks (likely cause of 2x/3x repeats)
   - many short fragments (single chars) that indicate fragmentation
   - per-protocol accumulator content (partial fragments)
  This function must be placed ABOVE UefiMain as requested.
*/
STATIC
VOID
DiagnosticCheckCaptureBuffers(VOID)
{
  TerminalPrintf(L"--- DIAGNOSTIC: Capture & Buffer Check ---\n");

  /* 1) Basic capture memory state */
  TerminalPrintf(L"[CAPMEM] active=%d allocated=%d\n",
                 gCaptureMemActive ? 1 : 0,
                 (gCaptureMem != NULL) ? 1 : 0);

  if (gCaptureMem) {
    TerminalPrintf(L"[CAPMEM] capacity lines=%u  chars/slot=%u  head=%u  count=%u\n",
                   (UINT32)gCaptureMaxLines, (UINT32)gCaptureMaxChars, (UINT32)gCaptureHead, (UINT32)gCaptureCount);

    /* compute basic statistics and look for consecutive duplicates and many very short lines */
    UINTN startIndex;
    if (gCaptureCount == 0) {
      TerminalPrintf(L"[CAPMEM] no lines currently captured\n");
    } else {
      if (gCaptureCount < gCaptureMaxLines) startIndex = 0;
      else startIndex = gCaptureHead; /* oldest entry */
      UINTN total = gCaptureCount;
      UINTN nonEmpty = 0;
      UINTN shortLines = 0;
      UINTN maxDupStreak = 1;
      UINTN currDupStreak = 0;
      /* compare with previous to detect streaks */
      CHAR16 *prev = NULL;
      /* Print up to first 10 non-empty sample lines */
      UINTN printedSample = 0;

      for (UINTN i = 0; i < total; ++i) {
        UINTN idx = (startIndex + i) % gCaptureMaxLines;
        CHAR16 *slot = gCaptureMem + idx * gCaptureSlotChars;
        UINTN len = slot ? StrLen(slot) : 0;
        if (len == 0) {
          /* skip */
          prev = NULL;
          currDupStreak = 0;
          continue;
        }
        nonEmpty++;
        if (len <= 3) shortLines++;

        /* duplicate detection */
        if (prev && StrCmp(prev, slot) == 0) {
          currDupStreak++;
        } else {
          if (currDupStreak > maxDupStreak) maxDupStreak = currDupStreak;
          currDupStreak = 1;
        }
        prev = slot;

        /* sample printing */
        if (printedSample < 10) {
          TerminalPrintf(L"[CAPMEM SAMPLE %u] len=%u: '", (UINT32)printedSample, (UINT32)len);
          /* print safely truncated sample */
          UINTN show = (len > 80) ? 80 : len;
          for (UINTN j = 0; j < show; ++j) Print(L"%c", slot[j]);
          if (len > show) Print(L"..."); 
          Print(L"'\n");
          printedSample++;
        }
      } /* for */

      if (currDupStreak > maxDupStreak) maxDupStreak = currDupStreak;

      TerminalPrintf(L"[CAPMEM] nonEmpty=%u  short(<=3)=%u  approx-max-consecutive-dup=%u\n",
                     (UINT32)nonEmpty, (UINT32)shortLines, (UINT32)maxDupStreak);
    }
  }

  /* 2) Inspect TextBuffer (the other ring) for fragmentation / duplicates */
  TerminalPrintf(L"[TEXTBUF] Lines stored=%u capacity=%u head=%u viewOffset=%d\n",
                 (UINT32)TextBufferCount, (UINT32)TextBufferLines, (UINT32)TextBufferHead, (INT32)ViewOffset);

  if (TextBufferCount > 0) {
    UINTN scan = (TextBufferCount > 200) ? 200 : TextBufferCount; /* limit scan */
    UINTN start = (TextBufferHead + (TextBufferCount - scan)) % TextBufferLines;
    CHAR16 *prev = NULL;
    UINTN maxDup = 1;
    UINTN dupStreak = 0;
    UINTN shortLines = 0;
    UINTN printed = 0;

    for (UINTN i = 0; i < scan; ++i) {
      UINTN idx = (start + i) % TextBufferLines;
      CHAR16 *line = TextBuffer[idx];
      UINTN len = line ? StrLen(line) : 0;
      if (len == 0) {
        prev = NULL;
        dupStreak = 0;
        continue;
      }
      if (len <= 3) shortLines++;
      if (prev && line && StrCmp(prev, line) == 0) {
        dupStreak++;
      } else {
        if (dupStreak > maxDup) maxDup = dupStreak;
        dupStreak = 1;
      }
      prev = line;

      if (printed < 10) {
        TerminalPrintf(L"[TEXT SAMPLE %u] len=%u: '", (UINT32)printed, (UINT32)len);
        UINTN show = (len > 80) ? 80 : len;
        for (UINTN j = 0; j < show; ++j) Print(L"%c", line[j]);
        if (len > show) Print(L"...");
        Print(L"'\n");
        printed++;
      }
    }
    if (dupStreak > maxDup) maxDup = dupStreak;
    TerminalPrintf(L"[TEXTBUF] scanned=%u  short(<=3)=%u  approx-max-consec-dup=%u\n",
                   (UINT32)scan, (UINT32)shortLines, (UINT32)maxDup);
  } else {
    TerminalPrintf(L"[TEXTBUF] empty\n");
  }

  /* 3) Inspect patched SimpleTextOut entries and their accumulators */
  TerminalPrintf(L"[PATCHED] count=%u  gOriginalOutputString=%p\n", (UINT32)gPatchedCount, (VOID*)gOriginalOutputString);
  if (gPatchedCount == 0) {
    TerminalPrintf(L"[PATCHED] No patched SimpleTextOut instances found\n");
  } else {
    /* check for repeated Originals among entries (could indicate multiple entries forwarding to same orig) */
    for (UINTN i = 0; i < gPatchedCount; ++i) {
      PATCHED_TEXTOUT_ENTRY *E = &gPatchedTextOuts[i];
      UINTN accumLen = (E->AccumBuf) ? StrLen(E->AccumBuf) : 0;
      TerminalPrintf(L"[PATCHED %u] Proto=%p Orig=%p AccumLen=%u AccumSample='",
                     (UINT32)i, (VOID*)E->Proto, (VOID*)E->OriginalOutputString, (UINT32)accumLen);
      if (accumLen > 0) {
        UINTN show = (accumLen > 80) ? 80 : accumLen;
        for (UINTN j = 0; j < show; ++j) Print(L"%c", E->AccumBuf[j]);
        if (accumLen > show) Print(L"..."); 
      }
      Print(L"'\n");
    }

    /* duplicate original pointer detection (simple n^2) */
    UINTN dupPairs = 0;
    for (UINTN i = 0; i < gPatchedCount; ++i) {
      for (UINTN j = i + 1; j < gPatchedCount; ++j) {
        if (gPatchedTextOuts[i].OriginalOutputString == gPatchedTextOuts[j].OriginalOutputString) {
          dupPairs++;
          TerminalPrintf(L"[PATCHED WARNING] entries %u and %u share same OriginalOutputString=%p\n",
                         (UINT32)i, (UINT32)j, (VOID*)gPatchedTextOuts[i].OriginalOutputString);
        }
      }
    }
    if (dupPairs == 0) TerminalPrintf(L"[PATCHED] no shared OriginalOutputString pointers detected\n");
  }

  /* 4) Quick consistency checks */
  if (gCaptureMem) {
    if (gCaptureHead >= gCaptureMaxLines) {
      TerminalPrintf(L"[ERROR] gCaptureHead (%u) >= gCaptureMaxLines (%u)\n", (UINT32)gCaptureHead, (UINT32)gCaptureMaxLines);
    }
    if (gCaptureCount > gCaptureMaxLines) {
      TerminalPrintf(L"[ERROR] gCaptureCount (%u) > gCaptureMaxLines (%u)\n", (UINT32)gCaptureCount, (UINT32)gCaptureMaxLines);
    }
  }

  TerminalPrintf(L"--- DIAGNOSTIC END ---\n");
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
    } else if (StrCmp(Cmd, L"savelog") == 0) {
		SaveConsoleBufferToMyLogs();
	} else if (StrCmp(Cmd, L"capmem") == 0) {
      DoCmdCapMem(Arg);
	} else if (StrCmp(Cmd, L"bugs_debug_mode") == 0) {
	  DiagnosticCheckCaptureBuffers();
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
  TerminalPrintf(L"  savelog			- test logging to USB device flash if USB_IO_PROTOCOL works \n");
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
  TerminalPrintf(L"  capmem start <lines> <chars> - allocate RAM capture (ring buffer) and start capture\n");
  TerminalPrintf(L"  capmem stop                  - stop capture (buffer kept)\n");
  TerminalPrintf(L"  capmem save                  - save captured memory buffer to \\EFI\\Boot\\myLogs\\\n");
  TerminalPrintf(L"  capmem free                  - free capture buffer\n");
  TerminalPrintf(L"  capmem status                - show capture status\n");
  TerminalPrintf(L"  bugs_debug_mode              - diagnostic function for duplicating lines in logs\n");
  

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