//
// Ac97Play.c - fixed version (no sin(), no unused locals)
// UEFI app: safe AC'97 mixer/controller reads and a tiny DMA playback test.
// Uses EFI_PCI_IO only (no IoLib).  Be careful: this will program DMA on the audio
// controller. Keep host/guest volume low or muted while testing.
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
#define NABM_POCTRL           0x0B  // byte : Transfer Control (bit0 = Run)

#pragma pack(push,1)
typedef struct {
  UINT32 BufferAddr;   // physical (device) address of sample buffer
  UINT16 SampleCount;  // number of 16-bit samples across channels
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

  Print(L"AC97Play (fixed): locate EFI_PCI_IO...\n");
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

  // Try to set sample rate (front DAC)
  {
    Status = IoWrite16BarIndex(PciIo, 0, MIXER_PCM_SAMPLE_RATE, (UINT16)48000);
    if (EFI_ERROR(Status)) Print(L"Write PCM sample rate failed: %r\n", Status);
    else Print(L"Wrote PCM sample rate = 48000\n");
  }

  // Prepare a tiny DMA playback test using BAR1 NABM
  Print(L"\nPreparing tiny AC'97 PCM OUT DMA test (single BDL entry)\n");

  const UINT32 sampleRate = 22050; // 48000
  const UINT32 seconds = 1;
  const UINT32 channels = 1; // 2 default - 1 mono
  const UINT32 frames = sampleRate * seconds;
  const UINT32 totalSamples = frames * channels;
  const UINTN  sampleBytes = (UINTN)totalSamples * sizeof(INT16);

  if (totalSamples > 0xFFFE) {
    Print(L"Buffer too large per single BDL entry; reduce seconds\n");
    goto CLEANUP;
  }

  {
    VOID *HostAudioBuf = NULL;
    VOID *HostBdl = NULL;
    EFI_PHYSICAL_ADDRESS DeviceAudioAddr = 0;
    EFI_PHYSICAL_ADDRESS DeviceBdlAddr = 0;
    VOID *audioMapToken = NULL;
    VOID *bdlMapToken = NULL;

    UINTN pagesAudio = EFI_SIZE_TO_PAGES(sampleBytes);
    UINTN pagesBdl = EFI_SIZE_TO_PAGES(sizeof(AC97_BDL_ENTRY) * 1);

    Print(L"Allocating audio buffer: %u samples (%u bytes), %u pages\n", (UINT32)totalSamples, (UINT32)sampleBytes, (UINT32)pagesAudio);

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

	// BEEEEEEEP
	
    // Fill audio buffer with integer square wave (1000 Hz) - no floating point
    {
      UINT32 freq = 1000;
      UINT32 samples_per_cycle = (freq > 0) ? (sampleRate / freq) : 1;
      if (samples_per_cycle == 0) samples_per_cycle = 1;
      UINT32 half = samples_per_cycle / 2;
      INT16 *samples = (INT16*)HostAudioBuf;
      UINT32 frame;
      for (frame = 0; frame < frames; ++frame) {
        INT16 sval = (INT16)((frame % samples_per_cycle) < half ? 0x6FFF : (INT16)-0x6FFF);
        // stereo: left then right
        samples[frame*2 + 0] = sval;
        samples[frame*2 + 1] = sval;
      }
    }
	

    Print(L"Audio buffer filled (square wave 1000Hz)\n");
	
	
/*	
	// --- Replace the original square-wave fill block with this melody generator ---
// NOTE struct (local)
typedef struct { UINT32 freq; UINT32 durMs; } NOTE;

// short Jingle Bells fragment (freq in Hz, duration in ms)
NOTE score[] = {
  {659, 250}, {659, 250}, {659, 500},    // E E E
  {659, 250}, {659, 250}, {659, 500},    // E E E
  {659, 250}, {784, 250}, {523, 250},    // E G C
  {587, 250}, {659, 750}, {0,   250}     // D E (rest)
};
UINT32 noteCount = sizeof(score) / sizeof(score[0]);

{
  INT16 *samples = (INT16*)HostAudioBuf;   // host buffer previously allocated
  UINT32 curFrame = 0;
  UINT32 n;

  for (n = 0; n < noteCount && curFrame < frames; ++n) {
    UINT32 freq = score[n].freq;
    UINT32 durMs = score[n].durMs;

    // frames for this note
    UINT32 framesThisNote = (durMs * sampleRate) / 1000;
    if (framesThisNote == 0) framesThisNote = 1;

    // clamp to remaining space
    if (curFrame + framesThisNote > frames) {
      framesThisNote = frames - curFrame;
    }

    // samples per cycle (integer); guard against 0
    UINT32 samples_per_cycle = 1;
    if (freq > 0) {
      samples_per_cycle = (freq <= sampleRate) ? (sampleRate / freq) : 1;
      if (samples_per_cycle == 0) samples_per_cycle = 1;
    }
    UINT32 half = samples_per_cycle / 2;

    // generate square wave (or silence if freq == 0)
    UINT32 f;
    for (f = 0; f < framesThisNote; ++f) {
      INT16 sval;
      if (freq == 0) {
        sval = 0; // rest
      } else {
        sval = ((f % samples_per_cycle) < half) ? 0x6FFF : (INT16)-0x6FFF;
      }

      // write to all channels (interleaved)
      UINT32 ch;
      for (ch = 0; ch < channels; ++ch) {
        samples[(curFrame + f) * channels + ch] = sval;
      }
    }

    curFrame += framesThisNote;
  }

  // zero-fill any remaining frames
  if (curFrame < frames) {
    UINT32 rem;
    for (rem = curFrame; rem < frames; ++rem) {
      UINT32 ch;
      for (ch = 0; ch < channels; ++ch) {
        samples[rem * channels + ch] = 0;
      }
    }
  }

  Print(L"Audio buffer filled (melody)\n");
}

	Print(L"Audio buffer filled (ok)\n");
	*/

    // Map audio buffer for device (Bus master DMA)
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

    // Prepare BDL entry
    AC97_BDL_ENTRY *bdl = (AC97_BDL_ENTRY*)HostBdl;
    bdl[0].BufferAddr = (UINT32)DeviceAudioAddr;
    bdl[0].SampleCount = (UINT16)totalSamples;
    bdl[0].Control = (UINT16)(0x4000); // last entry

    // Map the BDL
    {
      UINTN NumberOfBytes = sizeof(AC97_BDL_ENTRY) * 1;
      EFI_PHYSICAL_ADDRESS DeviceAddrBdl = 0;
      VOID *MapToken = NULL;
      Status = PciIo->Map(PciIo, EfiPciIoOperationBusMasterCommonBuffer, HostBdl, &NumberOfBytes, &DeviceAddrBdl, &MapToken);
      if (EFI_ERROR(Status)) {
        Print(L"PciIo->Map(BDL) failed: %r\n", Status);
        goto UNMAP_AUDIO;
      }
      DeviceBdlAddr = DeviceAddrBdl;
      bdlMapToken = MapToken;
      Print(L"BDL host VA=%p mapped device addr=0x%lx\n", HostBdl, (UINT64)DeviceBdlAddr);
    }

    // Program controller (POBDBAR, LVI, clear status, start)
    {
      UINT32 deviceAddr32 = (UINT32)DeviceBdlAddr;
      Status = IoWrite32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, deviceAddr32);
      if (EFI_ERROR(Status)) { Print(L"Write POBDBAR failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Wrote POBDBAR = 0x%08x\n", deviceAddr32);

      UINT8 lvi = 0;
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POLVI, 1, &lvi);
      if (EFI_ERROR(Status)) { Print(L"Write POLVI failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Wrote POLVI (LVI) = %u\n", lvi);

      // Clear status word at +0x06
      {
        UINT16 clr = 0x1C;
        Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + 0x06, 1, &clr);
        if (!EFI_ERROR(Status)) Print(L"Cleared NATX status (0x06) := 0x%04x\n", clr);
      }

      // Start transfer (set run bit)
      UINT8 ctrl = 0x01;
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      if (EFI_ERROR(Status)) { Print(L"Write POCTRL failed: %r\n", Status); goto UNMAP_BDL; }
      Print(L"Started PCM OUT DMA (Transfer Control set = 0x%02x)\n", ctrl);

      gBS->Stall((seconds + 1) * 1000000);

      // Stop DMA
      ctrl = 0x00;
      Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
      if (!EFI_ERROR(Status)) Print(L"Stopped PCM OUT DMA (Transfer Control cleared)\n");
    }

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
      PciIo->FreeBuffer(PciIo, EFI_SIZE_TO_PAGES(sizeof(AC97_BDL_ENTRY) * 1), HostBdl);
      HostBdl = NULL;
    }
FREE_AUDIO:
    if (HostAudioBuf) {
      PciIo->FreeBuffer(PciIo, EFI_SIZE_TO_PAGES(sampleBytes), HostAudioBuf);
      HostAudioBuf = NULL;
    }
  }

CLEANUP:
  if (Handles) FreePool(Handles);
  Print(L"\nDone. Press any key to exit.\n");
  return EFI_SUCCESS;
}
