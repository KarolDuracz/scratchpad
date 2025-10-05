//
// DumpPciBar.c
// UEFI app: dump PCI config for 00:05:00, print command register, BARs,
// then attempt correct reads using EFI_PCI_IO (Io.Read for I/O, Mem.Read for MMIO).
// If Command register disables I/O/Mem, optionally enable them.
// Build with EDK II (no IoLib required).
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

STATIC
VOID
DumpHexLine(UINT32 Offset, UINT32 Value)
{
  Print (L"  +0x%02x : 0x%08x\n", Offset, Value);
}

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE ImageHandle,
  IN EFI_SYSTEM_TABLE *SystemTable
  )
{
  EFI_STATUS              Status;
  EFI_HANDLE             *HandleBuffer = NULL;
  UINTN                   HandleCount = 0, Index;
  EFI_PCI_IO_PROTOCOL    *PciIo = NULL;
  UINTN                   Segment, Bus, Device, Function;
  BOOLEAN                 Found = FALSE;

  Print (L"DumpPciBar: locating EFI_PCI_IO handles...\n");

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status)) {
    Print (L"LocateHandleBuffer failed: %r\n", Status);
    return Status;
  }

  for (Index = 0; Index < HandleCount; ++Index) {
    PciIo = NULL;
    Status = gBS->HandleProtocol(HandleBuffer[Index], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) continue;

    Status = PciIo->GetLocation(PciIo, &Segment, &Bus, &Device, &Function);
    if (EFI_ERROR(Status)) continue;

    if ((Segment == TARGET_SEGMENT) && (Bus == TARGET_BUS) && (Device == TARGET_DEVICE) && (Function == TARGET_FUNC)) {
      Found = TRUE;
      Print (L"Found PCI device at %u:%u.%u\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function);

      // Dump first 16 dwords of config (0x00..0x3C)
      Print (L"\nPCI config (0x00..0x3C):\n");
      for (UINT32 off = 0; off <= 0x3C; off += 4) {
        UINT32 val = 0;
        Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, off, 1, &val);
        if (EFI_ERROR(Status)) {
          Print (L"  Read cfg @0x%02x failed: %r\n", off, Status);
        } else {
          DumpHexLine((UINT32)off, val);
        }
      }

      // Read Command register (offset 0x04, low 16 bits)
      {
        UINT16 cmd = 0;
        Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint16, 0x04, 1, &cmd);
        if (EFI_ERROR(Status)) {
          Print (L"Failed to read Command reg: %r\n", Status);
        } else {
          Print (L"\nPCI Command register = 0x%04x\n", cmd);
          Print (L"  bit0 = I/O Space (%a)\n", (cmd & 0x1) ? "ENABLED" : "disabled");
          Print (L"  bit1 = Memory Space (%a)\n", (cmd & 0x2) ? "ENABLED" : "disabled");
          Print (L"  bit2 = Bus Master (%a)\n",   (cmd & 0x4) ? "ENABLED" : "disabled");

          // If disabled, try enabling (safe: set bits 0/1/2)
          if ((cmd & 0x7) != 0x7) {
            Print (L"\nEnabling I/O, Memory and Bus Master bits in Command register...\n");
            UINT16 newcmd = (UINT16)(cmd | 0x7);
            Status = PciIo->Pci.Write(PciIo, EfiPciIoWidthUint16, 0x04, 1, &newcmd);
            if (EFI_ERROR(Status)) {
              Print (L"  Failed to write new Command reg: %r\n", Status);
            } else {
              // re-read
              UINT16 cmd2 = 0;
              PciIo->Pci.Read(PciIo, EfiPciIoWidthUint16, 0x04, 1, &cmd2);
              Print (L"  Command now = 0x%04x\n", cmd2);
            }
          }
        }
      }

      // Inspect BARs 0..5 (cfg offsets 0x10..0x24)
      Print (L"\nBARs:\n");
      for (UINTN bar = 0; bar < 6; ++bar) {
        UINT32 off = 0x10 + (UINT32)(bar * 4);
        UINT32 barval = 0;
        Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, off, 1, &barval);
        if (EFI_ERROR(Status)) {
          Print (L"  BAR%u read failed: %r\n", (UINT32)bar, Status);
          continue;
        }

        if (barval == 0xFFFFFFFFu) {
          Print (L"  BAR%u: 0xFFFFFFFF (not implemented or inaccessible)\n", (UINT32)bar);
          continue;
        }

        // Determine type
        if (barval & 0x1) {
          UINT32 iobase = barval & ~0x3u;
          Print (L"  BAR%u: I/O space, raw=0x%08x, base=0x%08x\n", (UINT32)bar, barval, iobase);

          // Try to read 32-bit at offset 0 from I/O BAR using PciIo->Io.Read
          {
            UINT32 io_val = 0;
            Status = PciIo->Io.Read(PciIo, EfiPciIoWidthUint32, (UINT8)bar, 0, 1, &io_val);
            if (EFI_ERROR(Status)) {
              Print (L"    PciIo->Io.Read failed: %r\n", Status);
            } else {
              Print (L"    First dword at I/O BAR%u offset 0 = 0x%08x\n", (UINT32)bar, io_val);
            }
          }
        } else {
          // Memory BAR
          UINT32 type = (barval >> 1) & 0x3; // bit[3:1] type
          UINT64 base = (UINT64)(barval & ~0xFul);

          if (type == 0x2) {
            // 64-bit BAR: read high dword from next BAR
            UINT32 barhigh = 0;
            Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, off + 4, 1, &barhigh);
            if (!EFI_ERROR(Status)) {
              base |= ((UINT64)barhigh << 32);
            }
            Print (L"  BAR%u: MMIO (64-bit), raw=0x%08x, base=0x%016lx\n", (UINT32)bar, barval, base);
            bar++; // skip next (upper) BAR in loop
          } else {
            Print (L"  BAR%u: MMIO (32-bit), raw=0x%08x, base=0x%016lx\n", (UINT32)bar, barval, base);
          }

          // Try Mem.Read from bar index 'bar' at offset 0
          {
            UINT32 buf[4] = {0,0,0,0};
            EFI_PCI_IO_PROTOCOL_WIDTH Width = EfiPciIoWidthUint32;
            Status = PciIo->Mem.Read(PciIo, Width, (UINT8)bar, 0, 4, buf);
            if (EFI_ERROR(Status)) {
              Print (L"    PciIo->Mem.Read failed for BAR%u: %r\n", (UINT32)bar, Status);
            } else {
              Print (L"    First 4 dwords at BAR%u:\n", (UINT32)bar);
              for (UINTN i=0; i<4; ++i) {
                Print (L"      +0x%02x : 0x%08x\n", (UINT32)(i*4), buf[i]);
              }
            }
          }
        }
      } // for each BAR

      break; // found target, leave loop
    } // if location match
  } // for handles

  if (!Found) {
    Print (L"Target PCI device not found at 00:05:00\n");
  }

  if (HandleBuffer != NULL) {
    FreePool(HandleBuffer);
  }

  Print (L"\nDone. Press any key to exit.\n");
  return EFI_SUCCESS;
}
