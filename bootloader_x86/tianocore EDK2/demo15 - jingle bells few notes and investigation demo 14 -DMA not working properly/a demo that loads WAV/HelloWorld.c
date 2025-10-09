//
// Ac97Play_3voices_with_wav.c
// UEFI app: AC'97 DMA playback demo mixing 3 channels where channel 1 is a WAV file
// loaded from the same folder as the .efi image. Multi-BDL, reset handling included.
//
// WARNING: This programs DMA on the audio controller. Keep host/guest volume low or muted.
//

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DevicePathLib.h>        // ConvertDevicePathToText
#include <Protocol/PciIo.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/LoadedImage.h>
//#include <Guid/FileInfo.h>



// Local EFI_FILE_INFO GUID (09576E92-6D3F-11D2-8E39-00A0C969723B)
// Some EDKII trees do not provide Guid/FileInfo.h; define it locally.
STATIC EFI_GUID FileInfoGuid = { \
  0x09576E92, 0x6D3F, 0x11D2, \
  {0x8E, 0x39, 0x00, 0xA0, 0xC9, 0x69, 0x72, 0x3B} \
};

// --- Local EFI_FILE_INFO definition & GUID (for toolchains that lack Guid/FileInfo.h) ---
// Standard EFI_FILE_INFO GUID: 09576E92-6D3F-11D2-8E39-00A0C969723B
//STATIC CONST EFI_GUID FileInfoGuid = {
//  0x09576E92, 0x6D3F, 0x11D2, {0x8E,0x39,0x00,0xA0,0xC9,0x69,0x72,0x3B}
//};

// If EFI_FILE_INFO is not provided by the platform headers, provide a minimal
// compatible definition so we can read FileSize returned by File->GetInfo().
// UEFI defines three EFI_TIME fields between Size and FileSize; instead of re-
// defining EFI_TIME we reserve 48 bytes (3 * 16) which matches typical EFI_TIME size.
#ifndef EFI_FILE_INFO_DEFINED_BY_PLATFORM
#pragma pack(push,1)
typedef struct {
  UINT64  Size;               // size of this structure including variable filename
  UINT8   ReservedTimes[48];  // placeholder for Create/Access/Modification EFI_TIMEs
  UINT64  FileSize;           // actual file size in bytes
  UINT64  PhysicalSize;       // physical size on disk
  UINT32  Attribute;          // file attributes
  CHAR16  FileName[1];        // variable-length, null-terminated UTF16 filename
} EFI_FILE_INFO;
#pragma pack(pop)
#define EFI_FILE_INFO_DEFINED_BY_PLATFORM 1
#endif


#define TARGET_SEGMENT 0
#define TARGET_BUS     0
#define TARGET_DEVICE  5
#define TARGET_FUNC    0

// put your wav filename here (in the same folder as the .efi)
#define WAV_FILENAME L"jet.wav"

#define MIXER_RESET_REG       0x00
#define MIXER_MASTER_VOL      0x02
#define MIXER_PCM_OUT_VOL     0x18
#define MIXER_PCM_SAMPLE_RATE 0x2C

#define NABM_PCM_OUT_BASE_OFF 0x10
#define NABM_POBDBAR          0x00
#define NABM_POLVI            0x05
#define NABM_POCTRL           0x0B
#define NABM_STATUS_WORD      0x06

#pragma pack(push,1)
typedef struct {
  UINT32 BufferAddr;
  UINT16 SampleCount;
  UINT16 Control;
} AC97_BDL_ENTRY;
#pragma pack(pop)

// WAV helpers
typedef struct {
  UINT32 sampleRate;
  UINT16 channels;
  UINT16 bitsPerSample;
  UINT8 *data;       // pointer to sample bytes (start of 'data' chunk)
  UINT32 dataSize;   // bytes (not samples)
} WAV_FILE;

// helper I/O wrappers
STATIC
EFI_STATUS
IoWrite16BarIndex (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINT64              Offset,
  IN UINT16              Value
  )
{
  return PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, BarIndex, Offset, 1, &Value);
}

STATIC
EFI_STATUS
IoWrite32BarIndex (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINT64              Offset,
  IN UINT32              Value
  )
{
  return PciIo->Io.Write(PciIo, EfiPciIoWidthUint32, BarIndex, Offset, 1, &Value);
}

STATIC
EFI_STATUS
IoRead16BarIndex (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINT64              Offset,
  OUT UINT16            *Value
  )
{
  return PciIo->Io.Read(PciIo, EfiPciIoWidthUint16, BarIndex, Offset, 1, Value);
}

STATIC
EFI_STATUS
IoRead32BarIndex (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINT64              Offset,
  OUT UINT32            *Value
  )
{
  return PciIo->Io.Read(PciIo, EfiPciIoWidthUint32, BarIndex, Offset, 1, Value);
}

// Stop + Reset helper (ensures PCM OUT is in a clean state)
STATIC
EFI_STATUS
Ac97StopAndResetPcmOut(
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
  EFI_STATUS Status;
  UINT8 ctrl;
  UINTN retry;
  UINT8 readVal;
  UINT16 clr;

  // Clear RUN
  ctrl = 0x00;
  (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);

  // Assert RESET (bit1)
  ctrl = 0x02;
  Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
  if (!EFI_ERROR(Status)) {
    for (retry = 0; retry < 2000; ++retry) {
      Status = PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &readVal);
      if (EFI_ERROR(Status)) break;
      if ((readVal & 0x02) == 0) break;
      gBS->Stall(1000);
    }
  }

  // Clear NATX status
  clr = 0x1C;
  (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &clr);

  // Zero POBDBAR and LVI
  (VOID)IoWrite32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, 0);
  {
    UINT8 lzero = 0;
    (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POLVI, 1, &lzero);
  }

  return EFI_SUCCESS;
}

// Find the directory of the running image and open WAV_FILENAME inside it.
// Returns allocated buffer with file contents in *OutBuffer (must free with FreePool).
// OutSize receives file size in bytes.
STATIC
EFI_STATUS
OpenFileFromImageDir(
  IN  EFI_HANDLE         ImageHandle,
  IN  CHAR16            *FileName,
  OUT VOID             **OutBuffer,
  OUT UINTN             *OutSize
  )
{
  EFI_STATUS Status;
  EFI_LOADED_IMAGE_PROTOCOL *LoadedImage = NULL;
  EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *SimpleFs = NULL;
  EFI_FILE_PROTOCOL *Root = NULL;
  EFI_FILE_PROTOCOL *Dir = NULL;
  EFI_FILE_PROTOCOL *File = NULL;
  CHAR16 *dpText = NULL;
  CHAR16 *firstSlash = NULL;
  CHAR16 *lastSlash = NULL;
  CHAR16 *walk;
  CHAR16 tokenName[260];
  UINTN tokenLen;
  EFI_FILE_INFO *FileInfo = NULL;
  UINTN FileInfoSize = 0;
  UINTN ReadSize;
  VOID *Buffer = NULL;
  UINTN Size = 0;

  *OutBuffer = NULL;
  *OutSize = 0;

  // Get LoadedImage for this image
  Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
  if (EFI_ERROR(Status)) {
    Print(L"OpenFileFromImageDir: Failed to get LoadedImage: %r\n", Status);
    return Status;
  }

  // Get simple file system for the device that loaded image
  Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&SimpleFs);
  if (EFI_ERROR(Status)) {
    Print(L"OpenFileFromImageDir: Failed to get SimpleFileSystem: %r\n", Status);
    return Status;
  }

  // Open volume root
  Status = SimpleFs->OpenVolume(SimpleFs, &Root);
  if (EFI_ERROR(Status)) {
    Print(L"OpenFileFromImageDir: OpenVolume failed: %r\n", Status);
    return Status;
  }

  // Convert image device path to text so we can parse the file path portion
  dpText = ConvertDevicePathToText(LoadedImage->FilePath, TRUE, TRUE);
  if (dpText == NULL) {
    Print(L"OpenFileFromImageDir: ConvertDevicePathToText failed\n");
    Root->Close(Root);
    return EFI_NOT_FOUND;
  }
  
  
  
  // Status = Root->Open(Root, &File, L"\\path\\to\\jet.wav", EFI_FILE_MODE_READ, 0);


  // dpText generally contains backslash path portion like "\EFI\BOOT\APP.EFI"
  // find first backslash in text
  firstSlash = StrStr(dpText, L"\\");
  if (firstSlash == NULL) {
    // no path part: open root directly
    Dir = Root;
  } else {
    // find last backslash (where filename starts after)
    lastSlash = NULL;
    for (walk = firstSlash; *walk != L'\0'; ++walk) {
      if (*walk == L'\\') lastSlash = walk;
    }
    if (lastSlash == NULL) {
      // Odd; use root
      Dir = Root;
    } else {
      // Create/open directories token by token from root up to lastSlash (excluding)
      Dir = Root;
      CHAR16 *p = firstSlash + 1; // skip leading '\'
      while (p < lastSlash && *p != L'\0') {
        // extract token up to next backslash or lastSlash
        CHAR16 *q = p;
        tokenLen = 0;
        while (q < lastSlash && *q != L'\\') { tokenName[tokenLen++] = *q; q++; }
        tokenName[tokenLen] = L'\0';
        // open next directory (readonly)
        EFI_FILE_PROTOCOL *Next = NULL;
        Status = Dir->Open(Dir, &Next, tokenName, EFI_FILE_MODE_READ, 0);
        if (EFI_ERROR(Status)) {
          Print(L"OpenFileFromImageDir: Failed to open dir token '%s': %r\n", tokenName, Status);
          if (Dir != Root) Dir->Close(Dir);
          Root->Close(Root);
          FreePool(dpText);
          return Status;
        }
        // close previous Dir handle (except Root) and continue
        if (Dir != Root) Dir->Close(Dir);
        Dir = Next;
        // advance p
        p = q;
        if (*p == L'\\') ++p;
      } // token loop
    }
  }

  // Now Dir points at the directory where the image resides (or Root).
  // Open the WAV file inside this directory
  //Status = Dir->Open(Dir, &File, FileName, EFI_FILE_MODE_READ, 0);
  Print(L"Image loaded from handle device = %p (should be fs1:)\n", LoadedImage->DeviceHandle);

  Status = Dir->Open(Dir, &File, L"fs1:\\__bin\\a09-10-2025\\jet.wav", EFI_FILE_MODE_READ, 0); // <<<<<<<<<<<<<<<<<<<<<<<<<< hard code path to file on USB flash drive
  if (EFI_ERROR(Status)) {
    Print(L"OpenFileFromImageDir: Failed to open '%s' in image folder: %r\n", FileName, Status);
    if (Dir != Root) Dir->Close(Dir);
    Root->Close(Root);
    FreePool(dpText);
    return Status;
  }

  // Get file size via GetInfo
  FileInfoSize = 0;
    // old:
  // Status = File->GetInfo(File, &gEfiFileInfoGuid, &FileInfoSize, NULL);

			// Allocate buffer for EFI_FILE_INFO structure and retrieve file information
			//EFI_FILE_INFO *FileInfo = NULL; //

			FileInfo = AllocatePool(FileInfoSize);
			if (FileInfo == NULL) {
			  Print(L"Failed to allocate FileInfo buffer (size %u)\n", (UINT32)FileInfoSize);
			  return EFI_OUT_OF_RESOURCES;
			}

			// Now retrieve the file info structure from the file handle
			//Status = File->GetInfo(File, &FileInfoGuid, &FileInfoSize, FileInfo);
			Status = File->GetInfo(File, &FileInfoGuid, &FileInfoSize, NULL);
			if (EFI_ERROR(Status)) {
			  Print(L"File->GetInfo() failed: %r\n", Status);
			  FreePool(FileInfo);
			  return Status;
			}

			// At this point, FileInfo->FileSize is valid and can be used:
			UINTN FileSize = (UINTN) FileInfo->FileSize;
			Print(L"WAV file size = %u bytes\n", (UINT32)FileSize);

			// error C2220 -----> 
		  // new:
		  //Status = File->GetInfo(File, &FileInfoGuid, &FileInfoSize, NULL); /// ????? last arg is NULL or 

		 /* ????????????
		 Status = File->GetInfo(File, &FileInfoGuid, &FileInfoSize, FileInfo); // <<< NULL or FileInfo
		if (!EFI_ERROR(Status)) {
		  UINTN fileSize = (UINTN) FileInfo->FileSize;
		  ...
		}
		*/
  
  if (Status == EFI_BUFFER_TOO_SMALL) {
    FileInfo = AllocatePool(FileInfoSize);
    if (FileInfo == NULL) {
      Print(L"OpenFileFromImageDir: AllocatePool(FileInfo) failed\n");
      File->Close(File);
      if (Dir != Root) Dir->Close(Dir);
      Root->Close(Root);
      FreePool(dpText);
      return EFI_OUT_OF_RESOURCES;
    }
      // old:
  // Status = File->GetInfo(File, &gEfiFileInfoGuid, &FileInfoSize, FileInfo);

  // new:
  Status = File->GetInfo(File, &FileInfoGuid, &FileInfoSize, FileInfo);

  }
  if (EFI_ERROR(Status)) {
    Print(L"OpenFileFromImageDir: GetInfo failed: %r\n", Status);
    if (FileInfo) FreePool(FileInfo);
    File->Close(File);
    if (Dir != Root) Dir->Close(Dir);
    Root->Close(Root);
    FreePool(dpText);
    return Status;
  }

  Size = (UINTN)FileInfo->FileSize;
  if (Size == 0) {
    Print(L"OpenFileFromImageDir: file '%s' has size 0\n", FileName);
    FreePool(FileInfo);
    File->Close(File);
    if (Dir != Root) Dir->Close(Dir);
    Root->Close(Root);
    FreePool(dpText);
    return EFI_BAD_BUFFER_SIZE;
  }

  Buffer = AllocatePool(Size);
  if (Buffer == NULL) {
    Print(L"OpenFileFromImageDir: AllocatePool(file) failed\n");
    FreePool(FileInfo);
    File->Close(File);
    if (Dir != Root) Dir->Close(Dir);
    Root->Close(Root);
    FreePool(dpText);
    return EFI_OUT_OF_RESOURCES;
  }

  ReadSize = Size;
  Status = File->Read(File, &ReadSize, Buffer);
  if (EFI_ERROR(Status) || ReadSize != Size) {
    Print(L"OpenFileFromImageDir: File->Read failed: %r (read %u/%u)\n", Status, (UINT32)ReadSize, (UINT32)Size);
    FreePool(Buffer);
    FreePool(FileInfo);
    File->Close(File);
    if (Dir != Root) Dir->Close(Dir);
    Root->Close(Root);
    FreePool(dpText);
    return EFI_DEVICE_ERROR;
  }

  // done
  FreePool(FileInfo);
  File->Close(File);
  if (Dir != Root) Dir->Close(Dir);
  Root->Close(Root);
  FreePool(dpText);

  *OutBuffer = Buffer;
  *OutSize = Size;
  return EFI_SUCCESS;
}

// parse WAV from memory buffer (simple), fills WAV_FILE; data pointer points inside buffer
// Only supports PCM (format 1) and 16-bit samples (bitsPerSample==16).
STATIC
EFI_STATUS
ParseWavFromMemory(
  IN  VOID    *FileBuffer,
  IN  UINTN    FileSize,
  OUT WAV_FILE *OutWav
  )
{
  UINT8 *buf = (UINT8*)FileBuffer;
  UINTN pos = 0;

  ZeroMem(OutWav, sizeof(*OutWav));

  if (FileSize < 12) return EFI_INVALID_PARAMETER;
  // RIFF header
  if (buf[0] != 'R' || buf[1] != 'I' || buf[2] != 'F' || buf[3] != 'F') return EFI_INVALID_PARAMETER;
  // skip size
  if (buf[8] != 'W' || buf[9] != 'A' || buf[10] != 'V' || buf[11] != 'E') return EFI_INVALID_PARAMETER;
  pos = 12;

  // iterate chunks to find 'fmt ' and 'data'
  UINT16 audioFormat = 0;
  UINT32 fmtChannels = 0;
  UINT32 fmtSampleRate = 0;
  UINT16 fmtBits = 0;
  UINT8 *dataPtr = NULL;
  UINT32 dataSize = 0;

  while (pos + 8 <= FileSize) {
    char id[5]; id[4] = 0;
    id[0] = (char)buf[pos+0]; id[1] = (char)buf[pos+1]; id[2] = (char)buf[pos+2]; id[3] = (char)buf[pos+3];
    UINT32 chunkSize = *(UINT32*)(buf + pos + 4);
    // ensure little-endian safe
    // (on UEFI x86/x64 this direct deref is OK; if not, you'd perform read-bytes)
    pos += 8;
    if (pos + chunkSize > FileSize) {
      // invalid chunk size
      return EFI_INVALID_PARAMETER;
    }
    if (id[0] == 'f' && id[1] == 'm' && id[2] == 't' && id[3] == ' ') {
      // parse fmt chunk (at least 16 bytes)
      if (chunkSize < 16) return EFI_INVALID_PARAMETER;
      audioFormat = *(UINT16*)(buf + pos + 0);
      fmtChannels = *(UINT16*)(buf + pos + 2);
      fmtSampleRate = *(UINT32*)(buf + pos + 4);
      // skip byteRate (4) and blockAlign (2)
      fmtBits = *(UINT16*)(buf + pos + 14);
    } else if (id[0] == 'd' && id[1] == 'a' && id[2] == 't' && id[3] == 'a') {
      dataPtr = buf + pos;
      dataSize = chunkSize;
      // we do not break because fmt might come after data in some files; but usually fmt is earlier
    }
    pos += chunkSize;
    // chunk sizes are WORD aligned; if odd add pad
    if ((chunkSize & 1) && pos < FileSize) pos++;
  }

  if (dataPtr == NULL) return EFI_NOT_FOUND;
  if (audioFormat != 1) {
    Print(L"WAV: unsupported audio format %u (only PCM=1 supported)\n", audioFormat);
    return EFI_UNSUPPORTED;
  }
  if (fmtBits != 16) {
    Print(L"WAV: unsupported bits-per-sample %u (only 16 supported)\n", fmtBits);
    return EFI_UNSUPPORTED;
  }
  if (fmtChannels == 0) return EFI_INVALID_PARAMETER;

  OutWav->sampleRate = fmtSampleRate;
  OutWav->channels = (UINT16)fmtChannels;
  OutWav->bitsPerSample = fmtBits;
  OutWav->data = dataPtr;
  OutWav->dataSize = dataSize;
  return EFI_SUCCESS;
}

// -------------------------------------------------------------
// Main program: combines WAV file as voice 1 with 2 other square voices
// -------------------------------------------------------------
EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE *SystemTable
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0, Index;
  EFI_PCI_IO_PROTOCOL *PciIo = NULL;
  UINTN Seg, Bus, Dev, Func;
  BOOLEAN Found = FALSE;

  Print(L"AC97Play (3-voices + WAV): locate EFI_PCI_IO...\n");
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR(Status)) {
    Print(L"LocateHandleBuffer failed: %r\n", Status);
    return Status;
  }

  for (Index = 0; Index < HandleCount; ++Index) {
    Status = gBS->HandleProtocol(Handles[Index], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) continue;
    Status = PciIo->GetLocation(PciIo, &Seg, &Bus, &Dev, &Func);
    if (EFI_ERROR(Status)) continue;
    if (Seg == TARGET_SEGMENT && Bus == TARGET_BUS && Dev == TARGET_DEVICE && Func == TARGET_FUNC) {
      Found = TRUE;
      Print(L"Found target PCI device at %u:%u.%u\n", (UINT32)Bus, (UINT32)Dev, (UINT32)Func);
      break;
    }
  }
  if (!Found) {
    Print(L"Target PCI device not found (00:05:00)\n");
    if (Handles) FreePool(Handles);
    return EFI_NOT_FOUND;
  }

  // Show a config dword
  {
    UINT32 header0 = 0;
    PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x00, 1, &header0);
    Print(L"PCI header dword0 = 0x%08x\n", header0);
  }

  // Unmute & set PCM OUT vol
  {
    UINT16 master = 0;
    (VOID)IoRead16BarIndex(PciIo, 0, MIXER_MASTER_VOL, &master);
    (VOID)IoWrite16BarIndex(PciIo, 0, MIXER_MASTER_VOL, (UINT16)(master & ~(1u << 15)));
    (VOID)IoWrite16BarIndex(PciIo, 0, MIXER_PCM_OUT_VOL, 0x0000);
  }

  // Load WAV file from the same folder as this .efi
  VOID *wavBuf = NULL;
  UINTN wavSize = 0;
  WAV_FILE wav = {0};
  Status = OpenFileFromImageDir(ImageHandle, WAV_FILENAME, &wavBuf, &wavSize);
  if (EFI_ERROR(Status)) {
    Print(L"Warning: could not open WAV '%s' from image folder: %r\n", WAV_FILENAME, Status);
    // we'll still play pure synthesized voices if WAV not found
  } else {
    Print(L"WAV file '%s' loaded: %u bytes\n", WAV_FILENAME, (UINT32)wavSize);
    Status = ParseWavFromMemory(wavBuf, wavSize, &wav);
    if (EFI_ERROR(Status)) {
      Print(L"WAV parse failed: %r\n", Status);
      FreePool(wavBuf);
      wavBuf = NULL;
      ZeroMem(&wav, sizeof(wav));
    } else {
      Print(L"WAV parsed: sampleRate=%u channels=%u bits=%u dataBytes=%u\n",
            wav.sampleRate, wav.channels, wav.bitsPerSample, wav.dataSize);
    }
  }

  // Playback parameters
  const UINT32 sampleRate = 22050;
  const UINT32 channels = 2;
  const UINT32 MAX_SAMPLES_PER_ENTRY = 0xFFFE;
  const UINT32 totalDurationMs = 10000; // 10 seconds

  UINT32 frames = (sampleRate * totalDurationMs + 999) / 1000;
  UINT64 totalSamples64 = (UINT64)frames * (UINT64)channels;
  if (totalSamples64 > 0xFFFFFFFFULL) {
    Print(L"Too many samples\n");
    goto CLEANUP;
  }
  UINT32 totalSamples = (UINT32)totalSamples64;
  UINTN sampleBytes = (UINTN)totalSamples * sizeof(INT16);

  UINT32 numBdl = (totalSamples + (MAX_SAMPLES_PER_ENTRY - 1)) / MAX_SAMPLES_PER_ENTRY;
  if (numBdl == 0) numBdl = 1;
  if (numBdl > 256) {
    Print(L"Too many BDL entries required (%u)\n", numBdl);
    goto CLEANUP;
  }

  Print(L"Preparing mixed output: sampleRate=%u frames=%u samples=%u bytes=%u bdl=%u\n",
        sampleRate, frames, totalSamples, (UINT32)sampleBytes, numBdl);

  // voice amplitude settings
  const INT32 VOICE_AMP = 0x1400;    // amplitude for generated voices (reduced to leave headroom)
  const INT32 WAV_MIX_AMP = 0x1800;  // how strongly WAV contributes (scale down WAV sample to this amp)

  // define simple scores for voice 2 & 3 (as before)
  typedef struct { UINT32 freq; UINT32 durMs; } NOTE;
  NOTE score2[] = {
    {784,250}, {784,250}, {784,500},
    {784,250}, {784,250}, {784,500},
    {784,250}, {987,250}, {659,250},
    {740,250}, {784,750}, {0,250}
  };
  NOTE score3[] = {
    {329,250}, {329,250}, {329,500},
    {329,250}, {329,250}, {329,500},
    {329,250}, {392,250}, {262,250},
    {293,250}, {329,750}, {0,250}
  };

  {
    VOID *HostAudioBuf = NULL;
    VOID *HostBdl = NULL;
    EFI_PHYSICAL_ADDRESS DeviceAudioAddr = 0;
    EFI_PHYSICAL_ADDRESS DeviceBdlAddr = 0;
    VOID *audioMapToken = NULL;
    VOID *bdlMapToken = NULL;

    UINTN pagesAudio = EFI_SIZE_TO_PAGES(sampleBytes);
    UINTN pagesBdl = EFI_SIZE_TO_PAGES(sizeof(AC97_BDL_ENTRY) * (UINTN)numBdl);

    Status = PciIo->AllocateBuffer(PciIo, AllocateAnyPages, EfiBootServicesData, pagesAudio, &HostAudioBuf, 0);
    if (EFI_ERROR(Status) || HostAudioBuf == NULL) {
      Print(L"AllocateBuffer(audio) failed: %r\n", Status);
      goto CLEANUP;
    }
    Status = PciIo->AllocateBuffer(PciIo, AllocateAnyPages, EfiBootServicesData, pagesBdl, &HostBdl, 0);
    if (EFI_ERROR(Status) || HostBdl == NULL) {
      Print(L"AllocateBuffer(BDL) failed: %r\n", Status);
      goto FREE_AUDIO;
    }

    // Fill host audio buffer by mixing WAV (voice1) + voice2 + voice3
    {
      INT16 *samples = (INT16*)HostAudioBuf;
      // voice2/3 state
      UINT32 idx2 = 0, idx3 = 0;
      UINT32 framesLeft2 = 0, framesLeft3 = 0;
      UINT32 spc2 = 1, spc3 = 1; // samples per cycle
      UINT32 phase2 = 0, phase3 = 0;
      UINT32 len2 = sizeof(score2) / sizeof(score2[0]);
      UINT32 len3 = sizeof(score3) / sizeof(score3[0]);

      // init voice2 & voice3
      if (len2 > 0) {
        framesLeft2 = (score2[0].durMs * sampleRate + 999) / 1000; if (framesLeft2 == 0) framesLeft2 = 1;
        if (score2[0].freq > 0) spc2 = (score2[0].freq <= sampleRate) ? (sampleRate / score2[0].freq) : 1;
        phase2 = 0;
      }
      if (len3 > 0) {
        framesLeft3 = (score3[0].durMs * sampleRate + 999) / 1000; if (framesLeft3 == 0) framesLeft3 = 1;
        if (score3[0].freq > 0) spc3 = (score3[0].freq <= sampleRate) ? (sampleRate / score3[0].freq) : 1;
        phase3 = 0;
      }

      // WAV playback state: fixed-point source position and increment
      UINT64 wavFrameCount = 0; // number of source frames (samples per channel)
      if (wav.data != NULL) {
        // data bytes -> frames: (dataSize bytes) / (bytesPerFrame)
        UINT32 bytesPerFrame = (wav.channels * (wav.bitsPerSample / 8));
        if (bytesPerFrame == 0) bytesPerFrame = 1;
        wavFrameCount = wav.dataSize / bytesPerFrame;
      }
      UINT32 wavSrcPosFixed = 0;
      UINT32 wavIncrementFixed = 0; // 16.16 fixed point increment: (wavSampleRate / outputSampleRate) * (1<<16)
      if (wav.data != NULL && wav.sampleRate > 0) {
        // wavIncrementFixed = wav.sampleRate / sampleRate in 16.16
        wavIncrementFixed = (UINT32)(((UINT64)wav.sampleRate << 16) / (UINT64)sampleRate);
      }

      for (UINT32 frame = 0; frame < frames; ++frame) {
        INT32 mix = 0;

        // voice1 = WAV if present; otherwise simple square fallback
        if (wav.data != NULL && wavFrameCount > 0) {
          UINT32 srcIndex = wavSrcPosFixed >> 16; // integer frame index in source
          // wrap / loop if necessary (we loop to fill buffer)
          if (srcIndex >= wavFrameCount) {
            // wrap
            srcIndex %= (UINT32)wavFrameCount;
            wavSrcPosFixed = (srcIndex << 16);
          }
          INT32 wavSample = 0;
          // read sample(s)
          UINT8 *sp = (UINT8*)wav.data + (UINTN)srcIndex * (wav.channels * (wav.bitsPerSample / 8));
          if (wav.bitsPerSample == 16) {
            if (wav.channels == 1) {
              // mono 16-bit sample
              INT16 s = *(INT16*)sp;
              wavSample = (INT32)s;
            } else {
              // stereo: average L and R (interpret as interleaved little-endian)
              INT16 sL = *(INT16*)(sp + 0);
              INT16 sR = *(INT16*)(sp + 2);
              wavSample = ((INT32)sL + (INT32)sR) / 2;
            }
          }
          // scale WAV sample down to WAV_MIX_AMP amplitude range:
          // mix contribution = wavSample * WAV_MIX_AMP / 32768
          mix += (INT32)(((INT64)wavSample * (INT64)WAV_MIX_AMP) / 32768LL);
          // advance src pos
          wavSrcPosFixed += wavIncrementFixed;
          // if we reached end, wrap (loop)
          if ((wavSrcPosFixed >> 16) >= wavFrameCount) {
            if (wavFrameCount > 0) wavSrcPosFixed %= ((UINT32)wavFrameCount << 16);
          }
        } else {
          // fallback square wave if no WAV
          // simple tone at 659 Hz for voice1 as fallback
          static UINT32 fb_phase = 0;
          static UINT32 fb_spc = 1;
          if (fb_spc == 1) fb_spc = (659 <= sampleRate) ? (sampleRate / 659) : 1;
          UINT32 half = (fb_spc >> 1); if (half == 0) half = 1;
          INT32 sval = ((fb_phase % fb_spc) < half) ? VOICE_AMP : -VOICE_AMP;
          fb_phase = (fb_phase + 1) % (fb_spc ? fb_spc : 1);
          mix += sval;
        }

        // voice2
        if (len2 > 0) {
          if (framesLeft2 == 0) {
            idx2 = (idx2 + 1) % len2;
            framesLeft2 = (score2[idx2].durMs * sampleRate + 999) / 1000; if (framesLeft2 == 0) framesLeft2 = 1;
            if (score2[idx2].freq > 0) spc2 = (score2[idx2].freq <= sampleRate) ? (sampleRate / score2[idx2].freq) : 1;
            else spc2 = 1;
            phase2 = 0;
          }
          INT32 sval = 0;
          if (score2[idx2].freq == 0) sval = 0;
          else {
            UINT32 half = (spc2 >> 1); if (half == 0) half = 1;
            sval = ((phase2 % spc2) < half) ? VOICE_AMP : -VOICE_AMP;
            phase2 = (phase2 + 1) % (spc2 ? spc2 : 1);
          }
          mix += sval;
          framesLeft2--;
        }

        // voice3
        if (len3 > 0) {
          if (framesLeft3 == 0) {
            idx3 = (idx3 + 1) % len3;
            framesLeft3 = (score3[idx3].durMs * sampleRate + 999) / 1000; if (framesLeft3 == 0) framesLeft3 = 1;
            if (score3[idx3].freq > 0) spc3 = (score3[idx3].freq <= sampleRate) ? (sampleRate / score3[idx3].freq) : 1;
            else spc3 = 1;
            phase3 = 0;
          }
          INT32 sval = 0;
          if (score3[idx3].freq == 0) sval = 0;
          else {
            UINT32 half = (spc3 >> 1); if (half == 0) half = 1;
            sval = ((phase3 % spc3) < half) ? VOICE_AMP : -VOICE_AMP;
            phase3 = (phase3 + 1) % (spc3 ? spc3 : 1);
          }
          mix += sval;
          framesLeft3--;
        }

        // clamp mix to int16
        if (mix > 32767) mix = 32767;
        if (mix < -32768) mix = -32768;
        INT16 out = (INT16)mix;

        // stereo interleaved
        UINTN baseIndex = (UINTN)frame * channels;
        samples[baseIndex + 0] = out;
        samples[baseIndex + 1] = out;
      } // end frame loop

      Print(L"Audio buffer filled (mixed WAV + 2 voices): frames=%u\n", frames);
    }

    // Write PCM sample rate to codec (so codec sample rate equals our output)
    (VOID)IoWrite16BarIndex(PciIo, 0, MIXER_PCM_SAMPLE_RATE, (UINT16)sampleRate);

    // Ensure clean engine before programming BDL
    Ac97StopAndResetPcmOut(PciIo);

    // Map audio buffer for device
    {
      UINTN NumberOfBytes = sampleBytes;
      EFI_PHYSICAL_ADDRESS DeviceAddress = 0;
      VOID *MapToken = NULL;
      Status = PciIo->Map(PciIo, EfiPciIoOperationBusMasterCommonBuffer, HostAudioBuf, &NumberOfBytes, &DeviceAddress, &MapToken);
      if (EFI_ERROR(Status)) {
        Print(L"PciIo->Map(audio) failed: %r\n", Status);
        goto FREE_BDL_HOST;
      }
      DeviceAudioAddr = DeviceAddress;
      audioMapToken = MapToken;
      Print(L"Audio host VA=%p mapped device addr=0x%lx (bytes mapped %u)\n", HostAudioBuf, (UINT64)DeviceAudioAddr, (UINT32)NumberOfBytes);
    }

    // Prepare BDL entries
    {
      AC97_BDL_ENTRY *bdl = (AC97_BDL_ENTRY*)HostBdl;
      UINT32 remainingSamples = totalSamples;
      UINT32 sampleOffset = 0;
      for (UINT32 i = 0; i < numBdl; ++i) {
        UINT32 thisSamples = (remainingSamples > MAX_SAMPLES_PER_ENTRY) ? MAX_SAMPLES_PER_ENTRY : remainingSamples;
        UINTN byteOffset = (UINTN)sampleOffset * sizeof(INT16);
        UINT32 devAddr32 = (UINT32)((UINT64)DeviceAudioAddr + (UINT64)byteOffset);
        bdl[i].BufferAddr = devAddr32;
        bdl[i].SampleCount = (UINT16)thisSamples;
        bdl[i].Control = (UINT16)((i == (numBdl - 1)) ? 0x4000 : 0x0000);
        remainingSamples -= thisSamples;
        sampleOffset += thisSamples;
        Print(L"  BDL[%u] -> dev=0x%08x bytes=%u samples=%u\n", i, devAddr32, (UINT32)(thisSamples * sizeof(INT16)), thisSamples);
      }
    }

    // Map BDL
    {
      UINTN NumberOfBytes = sizeof(AC97_BDL_ENTRY) * (UINTN)numBdl;
      EFI_PHYSICAL_ADDRESS DeviceAddrBdl = 0;
      VOID *MapToken = NULL;
      Status = PciIo->Map(PciIo, EfiPciIoOperationBusMasterCommonBuffer, HostBdl, &NumberOfBytes, &DeviceAddrBdl, &MapToken);
      if (EFI_ERROR(Status)) {
        Print(L"PciIo->Map(BDL) failed: %r\n", Status);
        goto UNMAP_AUDIO;
      }
      DeviceBdlAddr = DeviceAddrBdl;
      bdlMapToken = MapToken;
      Print(L"BDL host VA=%p mapped device addr=0x%lx (entries=%u)\n", HostBdl, (UINT64)DeviceBdlAddr, numBdl);
    }

    // Diagnostics before start
    {
      UINT8 ctrlVal = 0; UINT16 stVal = 0; UINT32 pobVal = 0;
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrlVal);
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &stVal);
      IoRead32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, &pobVal);
      Print(L"Before start: POCTRL=0x%02x STATUS=0x%04x POBDBAR=0x%08x\n", ctrlVal, stVal, pobVal);
    }

    // Program controller (POBDBAR, LVI, clear status, start)
    {
      UINT32 deviceAddr32 = (UINT32)DeviceBdlAddr;
      Status = IoWrite32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, deviceAddr32);
      if (EFI_ERROR(Status)) { Print(L"Write POBDBAR failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Wrote POBDBAR = 0x%08x\n", deviceAddr32);

      UINT8 lvi = (UINT8)(numBdl - 1);
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POLVI, 1, &lvi);
      if (EFI_ERROR(Status)) { Print(L"Write POLVI failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Wrote POLVI (LVI) = %u\n", lvi);

      // Clear NATX status
      {
        UINT16 clr = 0x1C;
        (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &clr);
        Print(L"Cleared NATX status\n");
      }

      // Start transfer (set run)
      UINT8 ctrl = 0x01;
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      if (EFI_ERROR(Status)) { Print(L"Write POCTRL failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Started PCM OUT DMA\n");

      // Stall for duration + margin
      UINT64 stallUs = (UINT64)totalDurationMs * 1000ULL + 500000ULL;
      if (stallUs / 1000000ULL > 0xFFFFFFFFULL) stallUs = 0xFFFFFFFFULL;
      gBS->Stall((UINTN)stallUs);

      // Stop
      ctrl = 0x00;
      (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      Print(L"Stopped PCM OUT DMA\n");
    }

    // Diagnostics after stop
    {
      UINT8 ctrlVal = 0; UINT16 stVal = 0; UINT32 pobVal = 0;
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrlVal);
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &stVal);
      IoRead32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, &pobVal);
      Print(L"After stop: POCTRL=0x%02x STATUS=0x%04x POBDBAR=0x%08x\n", ctrlVal, stVal, pobVal);
    }

    // cleanup/reset so subsequent runs work
    Ac97StopAndResetPcmOut(PciIo);

UNMAP_BDL:
    if (bdlMapToken) { PciIo->Unmap(PciIo, bdlMapToken); bdlMapToken = NULL; }
UNMAP_AUDIO:
    if (audioMapToken) { PciIo->Unmap(PciIo, audioMapToken); audioMapToken = NULL; }
FREE_BDL_HOST:
    if (HostBdl) { PciIo->FreeBuffer(PciIo, pagesBdl, HostBdl); HostBdl = NULL; }
FREE_AUDIO:
    if (HostAudioBuf) { PciIo->FreeBuffer(PciIo, pagesAudio, HostAudioBuf); HostAudioBuf = NULL; }
  }

CLEANUP:
  if (wavBuf) FreePool(wavBuf);
  if (Handles) FreePool(Handles);
  Print(L"\nDone. Press any key to exit.\n");
  return EFI_SUCCESS;
}
