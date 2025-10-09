//
// Ac97Play_3voices.c
// UEFI app: AC'97 DMA playback demo mixing 3 simultaneous melody channels into stereo.
// Uses EFI_PCI_IO, multi-BDL, and reset handling so repeated runs work.
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
  UINT32 BufferAddr;
  UINT16 SampleCount;
  UINT16 Control;
} AC97_BDL_ENTRY;
#pragma pack(pop)

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

// Stop + Reset helper: ensure PCM OUT is clean before/after runs
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
    // poll until hardware clears reset bit
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

// Simple voice definition: score of (freq, durMs)
typedef struct { UINT32 freq; UINT32 durMs; } NOTE;

// We'll provide three score arrays (melody, harmony, bass)
STATIC NOTE score1[] = { // main melody (Jingle Bells fragment)
  {659,250}, {659,250}, {659,500},
  {659,250}, {659,250}, {659,500},
  {659,250}, {784,250}, {523,250},
  {587,250}, {659,750}, {0,250}
};
STATIC NOTE score2[] = { // simple harmony (a 3rd above often)
  {784,250}, {784,250}, {784,500},
  {784,250}, {784,250}, {784,500},
  {784,250}, {987,250}, {659,250},
  {740,250}, {784,750}, {0,250}
};
STATIC NOTE score3[] = { // bass (lower octave)
  {329,250}, {329,250}, {329,500},
  {329,250}, {329,250}, {329,500},
  {329,250}, {392,250}, {262,250},
  {293,250}, {329,750}, {0,250}
};

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

  Print(L"AC97Play (3-voice mix): locate EFI_PCI_IO...\n");
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

  // show pci header dword
  {
    UINT32 header0 = 0;
    PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x00, 1, &header0);
    Print(L"PCI header dword0 = 0x%08x\n", header0);
  }

  // Read/print some mixer regs
  {
    UINT16 r;
    (VOID)IoRead16BarIndex(PciIo, 0, MIXER_RESET_REG, &r);
    Print(L"RESET/CAPS (0x00) = 0x%04x\n", r);
    (VOID)IoRead16BarIndex(PciIo, 0, MIXER_MASTER_VOL, &r);
    (VOID)IoRead16BarIndex(PciIo, 0, MIXER_PCM_OUT_VOL, &r);
  }

  // Unmute & set PCM volume
  {
    UINT16 master = 0;
    (VOID)IoRead16BarIndex(PciIo, 0, MIXER_MASTER_VOL, &master);
    UINT16 newmaster = (UINT16)(master & ~(1u << 15));
    (VOID)IoWrite16BarIndex(PciIo, 0, MIXER_MASTER_VOL, newmaster);
    (VOID)IoWrite16BarIndex(PciIo, 0, MIXER_PCM_OUT_VOL, 0x0000);
  }

  // Playback params
  const UINT32 sampleRate = 22050;
  const UINT32 channels = 2; // stereo interleaved
  const UINT32 MAX_SAMPLES_PER_ENTRY = 0xFFFE;
  const UINT32 totalDurationMs = 10000; // 10 seconds

  // compute frames / samples
  const UINT32 frames = (sampleRate * totalDurationMs + 999) / 1000;
  const UINT64 totalSamples64 = (UINT64)frames * (UINT64)channels;
  if (totalSamples64 > 0xFFFFFFFFULL) {
    Print(L"Too many samples\n");
    goto CLEANUP;
  }
  const UINT32 totalSamples = (UINT32)totalSamples64;
  const UINTN sampleBytes = (UINTN)totalSamples * sizeof(INT16);

  UINT32 numBdl = (totalSamples + (MAX_SAMPLES_PER_ENTRY - 1)) / MAX_SAMPLES_PER_ENTRY;
  if (numBdl == 0) numBdl = 1;
  if (numBdl > 256) {
    Print(L"Too many BDL entries required (%u)\n", numBdl);
    goto CLEANUP;
  }

  Print(L"Preparing 3-voice mix: sampleRate=%u frames=%u totalSamples=%u bytes=%u bdl=%u\n",
        sampleRate, frames, totalSamples, (UINT32)sampleBytes, numBdl);

  // amplitude per voice - choose safe amplitude so sum of 3 voices fits int16
  const INT32 VOICE_AMP = 0x2000; // 8192 -> 3*8192=24576 < 32767

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

    // Fill buffer by mixing 3 voices
    {
      INT16 *samples = (INT16*)HostAudioBuf;
      // per-voice state
      UINT32 idx1 = 0, idx2 = 0, idx3 = 0;
      UINT32 posInNote1 = 0, posInNote2 = 0, posInNote3 = 0;
      UINT32 framesLeft1 = 0, framesLeft2 = 0, framesLeft3 = 0;
      UINT32 samples_per_cycle1 = 1, samples_per_cycle2 = 1, samples_per_cycle3 = 1;
      UINT32 phase1 = 0, phase2 = 0, phase3 = 0;

      UINT32 score1Len = sizeof(score1) / sizeof(score1[0]);
      UINT32 score2Len = sizeof(score2) / sizeof(score2[0]);
      UINT32 score3Len = sizeof(score3) / sizeof(score3[0]);

      // initialize first notes
      if (score1Len > 0) {
        framesLeft1 = (score1[0].durMs * sampleRate + 999) / 1000; if (framesLeft1 == 0) framesLeft1 = 1;
        if (score1[0].freq > 0) samples_per_cycle1 = (score1[0].freq <= sampleRate) ? (sampleRate / score1[0].freq) : 1;
        phase1 = 0;
      }
      if (score2Len > 0) {
        framesLeft2 = (score2[0].durMs * sampleRate + 999) / 1000; if (framesLeft2 == 0) framesLeft2 = 1;
        if (score2[0].freq > 0) samples_per_cycle2 = (score2[0].freq <= sampleRate) ? (sampleRate / score2[0].freq) : 1;
        phase2 = 0;
      }
      if (score3Len > 0) {
        framesLeft3 = (score3[0].durMs * sampleRate + 999) / 1000; if (framesLeft3 == 0) framesLeft3 = 1;
        if (score3[0].freq > 0) samples_per_cycle3 = (score3[0].freq <= sampleRate) ? (sampleRate / score3[0].freq) : 1;
        phase3 = 0;
      }

      UINT32 frame = 0;
      while (frame < frames) {
        // if a voice exhausted its current note, advance it
        if (framesLeft1 == 0 && score1Len > 0) {
          idx1 = (idx1 + 1) % score1Len;
          UINT32 durMs = score1[idx1].durMs;
          framesLeft1 = (durMs * sampleRate + 999) / 1000; if (framesLeft1 == 0) framesLeft1 = 1;
          if (score1[idx1].freq > 0) samples_per_cycle1 = (score1[idx1].freq <= sampleRate) ? (sampleRate / score1[idx1].freq) : 1;
          else samples_per_cycle1 = 1;
          phase1 = 0;
          posInNote1 = 0;
        }
        if (framesLeft2 == 0 && score2Len > 0) {
          idx2 = (idx2 + 1) % score2Len;
          UINT32 durMs = score2[idx2].durMs;
          framesLeft2 = (durMs * sampleRate + 999) / 1000; if (framesLeft2 == 0) framesLeft2 = 1;
          if (score2[idx2].freq > 0) samples_per_cycle2 = (score2[idx2].freq <= sampleRate) ? (sampleRate / score2[idx2].freq) : 1;
          else samples_per_cycle2 = 1;
          phase2 = 0;
          posInNote2 = 0;
        }
        if (framesLeft3 == 0 && score3Len > 0) {
          idx3 = (idx3 + 1) % score3Len;
          UINT32 durMs = score3[idx3].durMs;
          framesLeft3 = (durMs * sampleRate + 999) / 1000; if (framesLeft3 == 0) framesLeft3 = 1;
          if (score3[idx3].freq > 0) samples_per_cycle3 = (score3[idx3].freq <= sampleRate) ? (sampleRate / score3[idx3].freq) : 1;
          else samples_per_cycle3 = 1;
          phase3 = 0;
          posInNote3 = 0;
        }

        // compute each voice sample (square wave)
        INT32 mix = 0;

        // voice1
        if (score1Len > 0) {
          UINT32 f = score1[idx1].freq;
          INT32 sval = 0;
          if (f == 0) sval = 0;
          else {
            UINT32 half = (samples_per_cycle1 >> 1);
            if (half == 0) half = 1;
            sval = ((phase1 % samples_per_cycle1) < half) ? VOICE_AMP : -VOICE_AMP;
            phase1 = (phase1 + 1) % (samples_per_cycle1 ? samples_per_cycle1 : 1);
          }
          mix += sval;
          framesLeft1--; posInNote1++;
        }

        // voice2
        if (score2Len > 0) {
          UINT32 f = score2[idx2].freq;
          INT32 sval = 0;
          if (f == 0) sval = 0;
          else {
            UINT32 half = (samples_per_cycle2 >> 1);
            if (half == 0) half = 1;
            sval = ((phase2 % samples_per_cycle2) < half) ? VOICE_AMP : -VOICE_AMP;
            phase2 = (phase2 + 1) % (samples_per_cycle2 ? samples_per_cycle2 : 1);
          }
          mix += sval;
          framesLeft2--; posInNote2++;
        }

        // voice3
        if (score3Len > 0) {
          UINT32 f = score3[idx3].freq;
          INT32 sval = 0;
          if (f == 0) sval = 0;
          else {
            UINT32 half = (samples_per_cycle3 >> 1);
            if (half == 0) half = 1;
            sval = ((phase3 % samples_per_cycle3) < half) ? VOICE_AMP : -VOICE_AMP;
            phase3 = (phase3 + 1) % (samples_per_cycle3 ? samples_per_cycle3 : 1);
          }
          mix += sval;
          framesLeft3--; posInNote3++;
        }

        // clamp mix to int16 range
        if (mix > 32767) mix = 32767;
        if (mix < -32768) mix = -32768;
        INT16 out = (INT16)mix;

        // write stereo interleaved (duplicate left/right)
        UINTN baseIndex = (UINTN)frame * channels;
        samples[baseIndex + 0] = out;
        samples[baseIndex + 1] = out;

        frame++;
      } // while frames

      Print(L"Audio buffer filled: frames=%u\n", frames);
    }

    // Write sample rate to codec
    (VOID)IoWrite16BarIndex(PciIo, 0, MIXER_PCM_SAMPLE_RATE, (UINT16)sampleRate);

    // Ensure PCM OUT clean
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

    // Prepare BDL entries using DeviceAudioAddr + offsets
    {
      AC97_BDL_ENTRY *bdl = (AC97_BDL_ENTRY*)HostBdl;
      UINT32 remainingSamples = totalSamples;
      UINT32 sampleOffset = 0; // 16-bit samples offset
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

    // Map BDL buffer
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
      UINT8 ctrlVal = 0;
      UINT16 stVal = 0;
      UINT32 pobVal = 0;
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrlVal);
      PciIo->Io.Read(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &stVal);
      IoRead32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, &pobVal);
      Print(L"Before start: POCTRL=0x%02x STATUS=0x%04x POBDBAR=0x%08x\n", ctrlVal, stVal, pobVal);
    }

    // Program controller
    {
      UINT32 deviceAddr32 = (UINT32)DeviceBdlAddr;
      Status = IoWrite32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, deviceAddr32);
      if (EFI_ERROR(Status)) { Print(L"Write POBDBAR failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Wrote POBDBAR = 0x%08x\n", deviceAddr32);

      UINT8 lvi = (UINT8)(numBdl - 1);
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POLVI, 1, &lvi);
      if (EFI_ERROR(Status)) { Print(L"Write POLVI failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Wrote POLVI (LVI) = %u\n", lvi);

      // clear natx status
      {
        UINT16 clr = 0x1C;
        (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &clr);
        Print(L"Cleared NATX status\n");
      }

      // start
      UINT8 ctrl = 0x01;
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      if (EFI_ERROR(Status)) { Print(L"Write POCTRL failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Started PCM OUT DMA\n");

      // wait until done (duration + margin)
      UINT64 stallUs = (UINT64)totalDurationMs * 1000ULL + 500000ULL;
      if (stallUs / 1000000ULL > 0xFFFFFFFFULL) stallUs = 0xFFFFFFFFULL;
      gBS->Stall((UINTN)stallUs);

      // stop
      ctrl = 0x00;
      (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      Print(L"Stopped PCM OUT DMA\n");
    }

    // diagnostics after stop
    {
      UINT8 ctrlVal = 0;
      UINT16 stVal = 0;
      UINT32 pobVal = 0;
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
  if (Handles) FreePool(Handles);
  Print(L"\nDone. Press any key to exit.\n");
  return EFI_SUCCESS;
}
