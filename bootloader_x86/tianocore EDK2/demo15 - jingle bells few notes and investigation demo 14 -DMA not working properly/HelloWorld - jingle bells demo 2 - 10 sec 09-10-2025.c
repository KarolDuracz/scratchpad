//
// Ac97Play_jingle_10s.c
// UEFI app: AC'97 mixer/controller DMA playback test with multi-BDL and reset handling.
// Plays a repeated "Jingle Bells" fragment for 10 seconds (stereo interleaved).
//
// WARNING: This programs DMA on the audio controller. Keep host/guest volume low or muted while testing.
//

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Protocol/PciIo.h>

#define TARGET_SEGMENT 0
#define TARGET_BUS     0
#define TARGET_DEVICE  5
#define TARGET_FUNC    0

// AC'97 BAR offsets (relative to BAR0 (mixer) and BAR1 (NABM))
#define MIXER_RESET_REG       0x00  // word
#define MIXER_MASTER_VOL      0x02  // word (bit15 = mute)
#define MIXER_PCM_OUT_VOL     0x18  // word
#define MIXER_EXT_CAPS        0x28  // word
#define MIXER_EXT_CTRL        0x2A  // word
#define MIXER_PCM_SAMPLE_RATE 0x2C  // word (front DAC sample rate)

#define NABM_PCM_OUT_BASE_OFF 0x10  // BAR1 + 0x10 is PCM OUT NABM box base
#define NABM_POBDBAR          0x00  // dword: Buffer Descriptor List Physical Address
#define NABM_POLVI            0x05  // byte : Last Valid Index (LVI)
#define NABM_POCTRL           0x0B  // byte : Transfer Control (bit0 = Run, bit1 = Reset)
#define NABM_STATUS_WORD      0x06  // word : NATX status bits

#pragma pack(push,1)
typedef struct {
  UINT32 BufferAddr;   // physical (device) address of sample buffer (32-bit for controller)
  UINT16 SampleCount;  // number of 16-bit samples across channels (max 0xFFFE)
  UINT16 Control;      // bit15 = interrupt, bit14 = last entry
} AC97_BDL_ENTRY;
#pragma pack(pop)

// helper write/read functions
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
IoRead32BarIndex (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINT64              Offset,
  OUT UINT32            *Value
  )
{
  return PciIo->Io.Read(PciIo, EfiPciIoWidthUint32, BarIndex, Offset, 1, Value);
}

// Stop + Reset helper: make sure the PCM OUT channel is in a clean state
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

  // 1) Clear Run bit (ensure not running)
  ctrl = 0x00;
  Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
  if (EFI_ERROR(Status)) {
    Print(L"Ac97: failed to clear RUN: %r\n", Status);
  } else {
    Print(L"Ac97: POCTRL RUN cleared\n");
  }

  // 2) Assert Reset bit (bit1 = 0x02)
  ctrl = 0x02;
  Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
  if (EFI_ERROR(Status)) {
    Print(L"Ac97: failed to write RESET: %r\n", Status);
  } else {
    // poll until reset clears (hardware should clear the reset bit)
    for (retry = 0; retry < 2000; ++retry) { // up to ~2s
      Status = PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &readVal);
      if (EFI_ERROR(Status)) break;
      if ((readVal & 0x02) == 0) break; // reset bit cleared
      gBS->Stall(1000); // 1 ms
    }
    if (EFI_ERROR(Status)) {
      Print(L"Ac97: reset poll read failed: %r\n", Status);
    } else {
      Print(L"Ac97: RESET cleared (poll tried %u times)\n", (UINT32)retry);
    }
  }

  // 3) Clear NATX status word (write ones for bits to clear)
  clr = 0x1C;
  Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &clr);
  if (!EFI_ERROR(Status)) Print(L"Ac97: cleared NATX status (0x06) := 0x%04x\n", clr);
  else Print(L"Ac97: failed clearing NATX status: %r\n", Status);

  // 4) Zero POBDBAR and LVI (safe cleanup so stale pointers don't remain)
  (VOID)IoWrite32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, 0);
  {
    UINT8 lzero = 0;
    (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POLVI, 1, &lzero);
  }
  Print(L"Ac97: POBDBAR & LVI zeroed\n");

  return EFI_SUCCESS;
}

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

  Print(L"AC97Play (jingle 10s): locate EFI_PCI_IO...\n");
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

  // show a config dword
  {
    UINT32 header0 = 0;
    PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x00, 1, &header0);
    Print(L"PCI header dword0 = 0x%08x\n", header0);
  }

  // Read mixer regs (BAR0 is I/O BAR index 0)
  Print(L"\nReading AC'97 mixer registers (BAR0 offsets):\n");
  {
    UINT16 r;
    Status = IoRead16BarIndex(PciIo, 0, MIXER_RESET_REG, &r);
    if (EFI_ERROR(Status)) { Print(L"  reset/caps read failed: %r\n", Status); }
    else Print(L"  RESET/CAPS (0x00) = 0x%04x\n", r);

    Status = IoRead16BarIndex(PciIo, 0, MIXER_MASTER_VOL, &r);
    if (!EFI_ERROR(Status)) Print(L"  MASTER VOL (0x02) = 0x%04x\n", r);

    Status = IoRead16BarIndex(PciIo, 0, MIXER_PCM_OUT_VOL, &r);
    if (!EFI_ERROR(Status)) Print(L"  PCM OUT VOL (0x18) = 0x%04x\n", r);

    Status = IoRead16BarIndex(PciIo, 0, MIXER_EXT_CAPS, &r);
    if (!EFI_ERROR(Status)) Print(L"  EXT CAPS (0x28) = 0x%04x\n", r);

    Status = IoRead16BarIndex(PciIo, 0, MIXER_PCM_SAMPLE_RATE, &r);
    if (!EFI_ERROR(Status)) Print(L"  PCM FREQ (0x2C) = 0x%04x\n", r);
  }

  // Unmute and set PCM volume (best-effort)
  {
    UINT16 master = 0;
    Status = IoRead16BarIndex(PciIo, 0, MIXER_MASTER_VOL, &master);
    if (EFI_ERROR(Status)) {
      Print(L"Cannot read master vol to unmute: %r\n", Status);
    } else {
      UINT16 newmaster = (UINT16)(master & ~(1u << 15)); // clear mute bit
      Status = IoWrite16BarIndex(PciIo, 0, MIXER_MASTER_VOL, newmaster);
      if (EFI_ERROR(Status)) Print(L"Failed to write master vol: %r\n", Status);
      else Print(L"Unmuted master vol (wrote 0x%04x)\n", newmaster);
    }
    // Set PCM OUT vol to a reasonable default (0x0000 is often maximum/unmuted)
    {
      UINT16 pv = 0x0000;
      Status = IoWrite16BarIndex(PciIo, 0, MIXER_PCM_OUT_VOL, pv);
      if (EFI_ERROR(Status)) Print(L"Failed to write PCM OUT vol: %r\n", Status);
      else Print(L"Wrote PCM OUT VOL = 0x%04x\n", pv);
    }
  }

  // Playback parameters
  const UINT32 sampleRate = 22050; // sample rate used to generate notes
  const UINT32 channels = 2;       // AC'97 expects stereo interleaved
  const UINT32 MAX_SAMPLES_PER_ENTRY = 0xFFFE; // per AC'97 BDL sample-count limit
  const UINT32 targetDurationMs = 10000; // 10 seconds total playback buffer

  // Jingle Bells fragment (freq in Hz, duration in ms)
  typedef struct { UINT32 freq; UINT32 durMs; } NOTE;
  NOTE score[] = {
    {659, 250}, {659, 250}, {659, 500},    // E E E
    {659, 250}, {659, 250}, {659, 500},    // E E E
    {659, 250}, {784, 250}, {523, 250},    // E G C
    {587, 250}, {659, 750}, {0,   250}     // D E (rest)
  };
  const UINT32 scoreLen = sizeof(score) / sizeof(score[0]);

  // compute frames and buffer sizes for the 10 second target
  const UINT32 frames = (sampleRate * targetDurationMs + 999) / 1000; // ceil
  const UINT64 totalSamples64 = (UINT64)frames * (UINT64)channels;   // 16-bit samples across channels
  if (totalSamples64 > (UINT64)0xFFFFFFFF) {
    Print(L"Total samples overflow - too large\n");
    goto CLEANUP;
  }
  const UINT32 totalSamples = (UINT32)totalSamples64;
  const UINTN sampleBytes = (UINTN)totalSamples * sizeof(INT16);

  // determine how many BDL entries required
  UINT32 numBdl = (totalSamples + (MAX_SAMPLES_PER_ENTRY - 1)) / MAX_SAMPLES_PER_ENTRY;
  if (numBdl == 0) numBdl = 1;
  if (numBdl > 256) {
    Print(L"Too many BDL entries required (%u) - reduce sample rate or duration.\n", numBdl);
    goto CLEANUP;
  }

  Print(L"\nPreparing AC'97 PCM OUT DMA test (10s jingle) - frames=%u samples=%u bytes=%u bdl=%u\n",
        frames, totalSamples, (UINT32)sampleBytes, numBdl);

  {
    VOID *HostAudioBuf = NULL;
    VOID *HostBdl = NULL;
    EFI_PHYSICAL_ADDRESS DeviceAudioAddr = 0;
    EFI_PHYSICAL_ADDRESS DeviceBdlAddr = 0;
    VOID *audioMapToken = NULL;
    VOID *bdlMapToken = NULL;

    UINTN pagesAudio = EFI_SIZE_TO_PAGES(sampleBytes);
    UINTN pagesBdl = EFI_SIZE_TO_PAGES(sizeof(AC97_BDL_ENTRY) * (UINTN)numBdl);

    // allocate audio buffer
    Status = PciIo->AllocateBuffer(PciIo, AllocateAnyPages, EfiBootServicesData, pagesAudio, &HostAudioBuf, 0);
    if (EFI_ERROR(Status) || HostAudioBuf == NULL) {
      Print(L"AllocateBuffer(audio) failed: %r\n", Status);
      goto CLEANUP;
    }

    // allocate host BDL (we will fill it after mapping audio)
    Status = PciIo->AllocateBuffer(PciIo, AllocateAnyPages, EfiBootServicesData, pagesBdl, &HostBdl, 0);
    if (EFI_ERROR(Status) || HostBdl == NULL) {
      Print(L"AllocateBuffer(BDL) failed: %r\n", Status);
      goto FREE_AUDIO;
    }

    // fill audio buffer by repeating the score until we hit frames (10s)
    {
      INT16 *samples = (INT16*)HostAudioBuf;
      UINT32 curFrame = 0;
      UINT32 idx = 0;
      while (curFrame < frames) {
        NOTE note = score[idx % scoreLen];
        idx++;
        UINT32 freq = note.freq;
        UINT32 durMs = note.durMs;
        UINT32 framesThisNote = (durMs * sampleRate + 999) / 1000;
        if (framesThisNote == 0) framesThisNote = 1;
        if (curFrame + framesThisNote > frames) framesThisNote = frames - curFrame;

        // compute square wave cycle
        UINT32 samples_per_cycle = 1;
        if (freq > 0) {
          samples_per_cycle = (freq <= sampleRate) ? (sampleRate / freq) : 1;
          if (samples_per_cycle == 0) samples_per_cycle = 1;
        }
        UINT32 half = samples_per_cycle / 2;

        for (UINT32 f = 0; f < framesThisNote; ++f) {
          INT16 sval;
          if (freq == 0) sval = 0;
          else sval = ((f % samples_per_cycle) < half) ? (INT16)0x6FFF : (INT16)-0x6FFF;

          // stereo interleaved (left then right)
          UINTN baseIndex = (UINTN)(curFrame + f) * channels;
          samples[baseIndex + 0] = sval;
          samples[baseIndex + 1] = sval;
        }
        curFrame += framesThisNote;
      }
      Print(L"Audio buffer filled (10s jingle) frames=%u\n", (UINT32)frames);
    }

    // write sample rate to codec
    {
      Status = IoWrite16BarIndex(PciIo, 0, MIXER_PCM_SAMPLE_RATE, (UINT16)sampleRate);
      if (EFI_ERROR(Status)) Print(L"Write PCM sample rate failed: %r\n", Status);
      else Print(L"Wrote PCM sample rate = %u\n", (UINT32)sampleRate);
    }

    // Ensure the channel is reset/clean before programming BDL
    Ac97StopAndResetPcmOut(PciIo);

    // Map audio buffer for device (Bus master Common Buffer)
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

    // Prepare BDL entries (in host memory) using device audio address base + offsets
    {
      AC97_BDL_ENTRY *bdl = (AC97_BDL_ENTRY*)HostBdl;
      UINT32 remainingSamples = totalSamples;
      UINT32 sampleOffset = 0; // in 16-bit sample units
      for (UINT32 i = 0; i < numBdl; ++i) {
        UINT32 thisSamples = (remainingSamples > MAX_SAMPLES_PER_ENTRY) ? MAX_SAMPLES_PER_ENTRY : remainingSamples;
        UINTN byteOffset = (UINTN)sampleOffset * sizeof(INT16);
        UINT32 devAddr32 = (UINT32)((UINT64)DeviceAudioAddr + (UINT64)byteOffset);
        bdl[i].BufferAddr = devAddr32;
        bdl[i].SampleCount = (UINT16)thisSamples;
        bdl[i].Control = (UINT16)((i == (numBdl - 1)) ? 0x4000 : 0x0000); // last-entry flag for final
        remainingSamples -= thisSamples;
        sampleOffset += thisSamples;
        Print(L"  BDL[%u] -> dev=0x%08x bytes=%u samples=%u\n", i, devAddr32, (UINT32)(thisSamples * sizeof(INT16)), thisSamples);
      }
    }

    // Map the BDL host buffer so controller can read it
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

    // Diagnostic read BEFORE start
    {
      UINT8 ctrlVal = 0;
      UINT16 stVal = 0;
      UINT32 pobVal = 0;
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

      UINT8 lvi = (UINT8)(numBdl - 1); // last valid index
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POLVI, 1, &lvi);
      if (EFI_ERROR(Status)) { Print(L"Write POLVI failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Wrote POLVI (LVI) = %u\n", lvi);

      // Clear NATX status word at +0x06 (write status bits to clear)
      {
        UINT16 clr = 0x1C;
        Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &clr);
        if (!EFI_ERROR(Status)) Print(L"Cleared NATX status (0x06) := 0x%04x\n", clr);
      }

      // Start transfer (set run bit)
      UINT8 ctrl = 0x01;
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      if (EFI_ERROR(Status)) { Print(L"Write POCTRL failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Started PCM OUT DMA (Transfer Control set = 0x%02x)\n", ctrl);

      // Stall until finished: wait targetDurationMs + margin (ms -> us)
      UINT64 stallUs = (UINT64)targetDurationMs * 1000ULL + 500000ULL;
      if (stallUs / 1000000ULL > 0xFFFFFFFFULL) stallUs = 0xFFFFFFFFULL;
      gBS->Stall((UINTN)stallUs);

      // Stop DMA
      ctrl = 0x00;
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      if (!EFI_ERROR(Status)) Print(L"Stopped PCM OUT DMA (Transfer Control cleared)\n");
    }

    // Diagnostic read AFTER stop
    {
      UINT8 ctrlVal = 0;
      UINT16 stVal = 0;
      UINT32 pobVal = 0;
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrlVal);
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &stVal);
      IoRead32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, &pobVal);
      Print(L"After stop: POCTRL=0x%02x STATUS=0x%04x POBDBAR=0x%08x\n", ctrlVal, stVal, pobVal);
    }

    // Reset/cleanup so subsequent runs work
    Ac97StopAndResetPcmOut(PciIo);

UNMAP_BDL:
    if (bdlMapToken) {
      PciIo->Unmap(PciIo, bdlMapToken);
      bdlMapToken = NULL;
    }
UNMAP_AUDIO:
    if (audioMapToken) {
      PciIo->Unmap(PciIo, audioMapToken);
      audioMapToken = NULL;
    }
FREE_BDL_HOST:
    if (HostBdl) {
      PciIo->FreeBuffer(PciIo, pagesBdl, HostBdl);
      HostBdl = NULL;
    }
FREE_AUDIO:
    if (HostAudioBuf) {
      PciIo->FreeBuffer(PciIo, pagesAudio, HostAudioBuf);
      HostAudioBuf = NULL;
    }
  }

CLEANUP:
  if (Handles) FreePool(Handles);
  Print(L"\nDone. Press any key to exit.\n");
  return EFI_SUCCESS;
}
