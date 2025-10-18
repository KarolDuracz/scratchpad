/** UsbDiagApp.c
 *
 * EDK II UEFI application to perform safe USB Host/Port/Device inspection.
 *
 * Behavior:
 *  - Scans all PCI devices and finds USB Host Controllers (PCI class 0x0C / subclass 0x03)
 *  - Classifies controller by Programming Interface (ProgIF)
 *  - Reads BAR0 (assumes BAR0 holds MMIO for HC registers) and reads capability/operational registers
 *  - Reports key registers for EHCI/OHCI/xHCI and root-hub port registers
 *  - Locates EFI_USB_IO_PROTOCOL handles and performs safe device descriptor reads
 *  - DOES NOT perform writes except when ENABLE_RISKY_WRITES == 1
 *
 * Compile-time safety switch:
 *    #define ENABLE_RISKY_WRITES 0   // default: read-only
 *
 * WARNING: enabling writes can change controller state, drop OS drivers, reset devices, etc.
 */

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/PrintLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Protocol/PciIo.h>
#include <Protocol/UsbIo.h>
#include <Protocol/LoadedImage.h>

#include <IndustryStandard/Usb.h> // safe include if present in your build environment

#define ENABLE_RISKY_WRITES 0   // Set to 1 ONLY if you understand the risks and test in VM

//
//
// Useful macros for PCI config offsets
//
#define PCI_CFG_OFFSET_CLASSCODE 0x0B  // class code (byte)
#define PCI_CFG_OFFSET_SUBCLASS  0x0A  // subclass (byte)
#define PCI_CFG_OFFSET_PROGIF    0x09  // progIF (byte)
#define PCI_CFG_OFFSET_BAR0      0x10  // BAR0 (dword)

//
// EHCI/OHCI/xHCI offsets/heuristics we will use (per specs and common implementations).
// We avoid aggressive writes; we only read registers reported by common sources.
//
// For EHCI (USB2): Capability registers at BAR + 0x0..CAPLENGTH-1, operational registers start at (BAR + CAPLENGTH)
// In op regs: USBCMD @ op + 0x00, USBSTS @ op + 0x04, USBINTR @ op + 0x08, FRINDEX @ op + 0x0C
// PeriodicListBase @ op + 0x10, ASYNCLISTADDR @ op + 0x14
// PORTSC region commonly starts at op + 0x44 (4 bytes per port)
//
#define EHCI_CAPLENGTH_OFFSET    0x00  // byte at BAR + 0x00 (caplength)
#define EHCI_OP_USBCMD           0x00
#define EHCI_OP_USBSTS           0x04
#define EHCI_OP_USBINTR          0x08
#define EHCI_OP_FRINDEX          0x0C
#define EHCI_OP_PERIODICLIST     0x10
#define EHCI_OP_ASYNCLIST        0x14
#define EHCI_OP_PORTSC_BASE      0x44  // port status/control base offset from operational base

//
// For OHCI: Common offsets observed in implementations and used in logs
// Root hub descriptor and port status registers often around BAR + 0x40..0x60
//
#define OHCI_REG_HCREVISION      0x00
#define OHCI_REG_HCCONTROL       0x04
#define OHCI_REG_HCSTATUS        0x08
#define OHCI_REG_HCINTSTATUS     0x0C
#define OHCI_REG_HCCA            0x18
#define OHCI_REG_RHDESCRIPTORA   0x44
#define OHCI_REG_RHDESCRIPTORB   0x48
#define OHCI_REG_RHSTATUS        0x4C
#define OHCI_REG_RHPORT_BASE     0x54

//
// For xHCI: capability length at BAR+0x00 and operational registers start at BAR+CAPLENGTH
// xHCI port registers start at op + 0x400 + 0x10*(port-1) (per xHCI spec)
//
#define XHCI_CAPLENGTH_OFFSET   0x00
#define XHCI_OP_USBCMD          0x00
#define XHCI_OP_USBSTS          0x04
#define XHCI_OP_CRCR            0x18
#define XHCI_OP_PORTREGS_BASE   0x400
#define XHCI_OP_PORTREG_STRIDE  0x10

//
// Utility wrappers to read PCI config and memory via EFI_PCI_IO_PROTOCOL
//
STATIC
EFI_STATUS
ReadPci8 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT32 Offset,
  OUT UINT8 *Value
  )
{
  return PciIo->Pci.Read(PciIo, EfiPciIoWidthUint8, Offset, 1, Value);
}

STATIC
EFI_STATUS
ReadPci32 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT32 Offset,
  OUT UINT32 *Value
  )
{
  return PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, Offset, 1, Value);
}

STATIC
EFI_STATUS
MemRead32 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8 BarIndex,
  IN UINT64 Offset,
  OUT UINT32 *Value
  )
{
  return PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, BarIndex, Offset, 1, Value);
}

STATIC
EFI_STATUS
MemRead8Block (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8 BarIndex,
  IN UINT64 Offset,
  IN UINTN  Count,
  OUT VOID  *Buffer
  )
{
  return PciIo->Mem.Read(PciIo, EfiPciIoWidthUint8, BarIndex, Offset, Count, Buffer);
}

#if ENABLE_RISKY_WRITES
STATIC
EFI_STATUS
MemWrite32 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT32 BarIndex,
  IN UINT64 Offset,
  IN UINT32 Value
  )
{
  return PciIo->Mem.Write(PciIo, EfiPciIoWidthUint32, BarIndex, Offset, 1, &Value);
}
#endif

//
// Print helper that tolerates long lines
//
#define Info(...)  Print(__VA_ARGS__); Print(L"\n")

//
// Interpret helper: prints a breakdown of portSC bits (EHCI-like) from a raw 32-bit value
//
STATIC
VOID
DumpPortSCBits (UINT32 PortSc)
{
  Info(L"  Raw PORTSC = 0x%08x", PortSc);
  Info(L"    CCS (Connect Status) = %d", (PortSc >> 0) & 1);
  Info(L"    PE  (Port Enabled)   = %d", (PortSc >> 2) & 1);
  Info(L"    PR  (Port Reset)     = %d", (PortSc >> 8) & 1);
  Info(L"    PO  (Port Owner)     = %d", (PortSc >> 13) & 1);
  Info(L"    Speed (bits)         = 0x%02x", (PortSc >> 10) & 0x7);
  Info(L"    Change bits (CSC/CEC/etc) = 0x%02x", (PortSc >> 21) & 0x7F);
}

//
// Inspect a PCI handle that appears to be a USB host controller
//
STATIC
VOID
InspectUsbController (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8 ClassCode,
  IN UINT8 SubClass,
  IN UINT8 ProgIf
  )
{
  EFI_STATUS Status;
  UINT32 Bar0;
  UINT8 CapLen;
  //UINT32 opVal;
  UINT64 OpBase; // offset relative to BAR index 0 for Mem.Read
  UINT32 Read32;
  CHAR16 *ctrlType = L"Unknown";

  Info(L"--- Found PCI USB controller Class=0x%02x SubClass=0x%02x ProgIf=0x%02x ---",
       ClassCode, SubClass, ProgIf);

  // Read BAR0 from config
  Status = ReadPci32(PciIo, PCI_CFG_OFFSET_BAR0, &Bar0);
  if (EFI_ERROR(Status)) {
    Info(L"Failed to read BAR0 (Status=%r)", Status);
    return;
  }

  // Note: BAR value returned from Pci.Read is the BAR value; for MMIO it's a base address.
  Info(L"BAR0 (raw) = 0x%08x  (we will access via PciIo->Mem.Read using BarIndex=0)", Bar0);

  //
  // Classify controller by ProgIf:
  //   commonly: ProgIf 0x10 = OHCI, 0x20 = EHCI, 0x30 = xHCI (many platforms vary)
  //
  if (ProgIf == 0x10) {
    ctrlType = L"OHCI (USB 1.x)";
  } else if (ProgIf == 0x20) {
    ctrlType = L"EHCI (USB 2.0)";
  } else if (ProgIf == 0x30 || ProgIf == 0x3) {
    // some BIOSes use different codes; assume xHCI for 0x30 or 0x3 in some systems
    ctrlType = L"xHCI (USB 3.x / unified)";
  } else {
    ctrlType = L"Unknown/Other HC";
  }
  Info(L"Controller type guess: %s", ctrlType);

  //
  // For EHCI and xHCI we first read the capability length byte at BAR + 0x00.
  // For OHCI we will sample the root hub descriptor registers (common offsets).
  //
  Status = MemRead8Block(PciIo, 0, 0x0, 1, &CapLen);
  if (EFI_ERROR(Status)) {
    Info(L"Failed to read capability length at BAR+0x0 (Status=%r)", Status);
    CapLen = 0;
  } else {
    Info(L"CAPLENGTH byte @ BAR+0x0 = 0x%02x", CapLen);
  }

  //
  // Compute operational base (relative offset in PciIo->Mem.Read calls).
  // If CapLen is zero or nonsensical, default to 0x00 (safe read-only).
  //
  OpBase = (UINT64)(CapLen ? CapLen : 0x00);
  Info(L"Operational register base (offset from BAR0) = 0x%lx", OpBase);

  //
  // Per-controller detailed reads
  //
  if (ProgIf == 0x20) { // EHCI
    Info(L">>> EHCI (USB2) operational register snapshot (read-only):");

    // USBCMD
    Status = MemRead32(PciIo, 0, OpBase + EHCI_OP_USBCMD, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" USBCMD = 0x%08x", Read32);
      // decode Run/Stop bit (bit 0)
      Info(L"  Run/Stop (Run) = %d", Read32 & 1);
    } else {
      Info(L" Failed read USBCMD");
    }

    // USBSTS
    Status = MemRead32(PciIo, 0, OpBase + EHCI_OP_USBSTS, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" USBSTS = 0x%08x", Read32);
      Info(L"  HCHalted = %d", (Read32 >> 12) & 1); // typical bit pos per spec
    } else {
      Info(L" Failed read USBSTS");
    }

    // ASYNCLISTADDR and PERIODICLISTBASE
    Status = MemRead32(PciIo, 0, OpBase + EHCI_OP_ASYNCLIST, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" ASYNCLISTADDR = 0x%08x", Read32);
      if (Read32 == 0) {
        Info(L"  NOTE: ASYNCLISTADDR == 0 (no async schedule programmed)");
      }
    }
    Status = MemRead32(PciIo, 0, OpBase + EHCI_OP_PERIODICLIST, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" PERIODICLISTBASE = 0x%08x", Read32);
      if (Read32 == 0) {
        Info(L"  NOTE: Periodic list base == 0 (periodic schedule not programmed)");
      }
    }

    // FRINDEX
    Status = MemRead32(PciIo, 0, OpBase + EHCI_OP_FRINDEX, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" FRINDEX = 0x%08x", Read32);
    }

    // Probe up to 32 ports (EHCI HCSPARAMS may give N ports but reading is safe)
    Info(L"Reading up to 32 PORTSC entries (opbase+0x44 + 4*(n-1)):");
    for (UINTN port = 1; port <= 32; ++port) {
      UINT64 portOffset = OpBase + EHCI_OP_PORTSC_BASE + 4 * (port - 1);
      Status = MemRead32(PciIo, 0, portOffset, &Read32);
      if (EFI_ERROR(Status)) {
        Info(L"  Port %02d: read error", port);
        continue;
      }
      Info(L"  Port %02d: PORTSC raw 0x%08x", port, Read32);
      DumpPortSCBits(Read32);
      // Interpret common problematic state: CCS==1 && PE==0 => connected but not enabled
      if ((Read32 & 1) && (((Read32 >> 2) & 1) == 0)) {
        Info(L"    -> Device present but Port Enabled==0 (enumeration may not have completed)");
      }
      // PortOwner check (bit 13)
      if (((Read32 >> 13) & 1) == 1) {
        Info(L"    -> PortOwner==1 (companion controller likely owns this port). Do not force.");
      }
    }

#if ENABLE_RISKY_WRITES
    //
    // Example risky action (disabled by default): attempt to set Run bit on USBCMD
    // WARNING: This may conflict with OS drivers. Only enabled if developer compiled with ENABLE_RISKY_WRITES.
    //
    UINT32 usbcmd;
    Status = MemRead32(PciIo, 0, OpBase + EHCI_OP_USBCMD, &usbcmd);
    if (!EFI_ERROR(Status)) {
      Info(L" (WRITE-ENABLED) Attempting to set Run bit in USBCMD (was 0x%08x)", usbcmd);
      usbcmd |= 1; // set Run
      Status = MemWrite32(PciIo, 0, OpBase + EHCI_OP_USBCMD, usbcmd);
      Info(L"  Write returned %r", Status);
    }
#endif

  } else if (ProgIf == 0x10) { // OHCI
    Info(L">>> OHCI (USB1.x) operational snapshot (read-only):");

    // Read a small block of registers so we can parse them
    UINT8 block[0x80];
    Status = MemRead8Block(PciIo, 0, 0x00, sizeof(block), block);
    if (EFI_ERROR(Status)) {
      Info(L"Failed to read OHCI register block");
    } else {
      UINT32 rhDescriptorA = *(UINT32*)(block + OHCI_REG_RHDESCRIPTORA);
      UINT32 rhDescriptorB = *(UINT32*)(block + OHCI_REG_RHDESCRIPTORB);
      UINT32 rhStatus = *(UINT32*)(block + OHCI_REG_RHSTATUS);
      Info(L" HcRevision (raw) = 0x%02x", block[0]);
      Info(L" HcControl (raw) = 0x%08x", *(UINT32*)(block + OHCI_REG_HCCONTROL));
      Info(L" HcCmdStatus (raw) = 0x%08x", *(UINT32*)(block + OHCI_REG_HCSTATUS));
      Info(L" RhDescriptorA = 0x%08x", rhDescriptorA);
      Info(L" RhDescriptorB = 0x%08x", rhDescriptorB);
      Info(L" RhStatus      = 0x%08x", rhStatus);

      // Determine number of ports: RhDescriptorB low byte or fallback to 32 if weird
      UINTN nPorts = (rhDescriptorB & 0xFF);
      if (nPorts == 0 || nPorts > 32) {
        nPorts = 32;
      }
      Info(L" OHCI reports NPorts = %d (will probe)", nPorts);

      for (UINTN p = 1; p <= nPorts; ++p) {
        UINTN offset = OHCI_REG_RHPORT_BASE + 4 * (p - 1);
        UINT32 rhps = *(UINT32*)(block + offset);
        Info(L"  Port %02d RHPS (raw) = 0x%08x", p, rhps);
        Info(L"    CCS=%d PES=%d PR=%d", rhps & 1, (rhps >> 2) & 1, (rhps >> 8) & 1);
      }
    }

#if ENABLE_RISKY_WRITES
    // Example risky OHCI action (disabled by default): writing PR (port reset) to root port.
    Info(L" (WRITE-ENABLED) Example: to reset port 1 you'd write PR bit to BAR+0x54; disabled by default.");
#endif

  } else { // xHCI or unknown
    Info(L">>> xHCI/Other controller snapshot (read-only):");
    // Read capability length again (we did above) and sample operational registers
    Status = MemRead32(PciIo, 0, OpBase + XHCI_OP_USBCMD, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" USBCMD (op+0x00) = 0x%08x", Read32);
    } else {
      Info(L" Could not read USBCMD at computed opbase");
    }
    Status = MemRead32(PciIo, 0, OpBase + XHCI_OP_USBSTS, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" USBSTS (op+0x04) = 0x%08x", Read32);
      Info(L"  Halted = %d", (Read32 & 1));
    }
    // sample CRCR
    Status = MemRead32(PciIo, 0, OpBase + XHCI_OP_CRCR, &Read32);
    if (!EFI_ERROR(Status)) {
      Info(L" CRCR (Command Ring Control) (op+0x18) = 0x%08x", Read32);
    }
    // Probe first 16 ports using xHCI port register formula but be cautious
    Info(L"Reading xHCI port registers op+0x400 + stride 0x10 (first 16 ports):");
    for (UINTN p = 1; p <= 16; ++p) {
      UINT64 portOff = OpBase + XHCI_OP_PORTREGS_BASE + XHCI_OP_PORTREG_STRIDE * (p - 1);
      Status = MemRead32(PciIo, 0, portOff, &Read32);
      if (EFI_ERROR(Status)) {
        Info(L"  Port %02d: read error", p);
      } else {
        Info(L"  Port %02d PORTSC = 0x%08x", p, Read32);
        DumpPortSCBits(Read32);
      }
    }
  }

  //
  // Now check whether there are EFI_USB_IO_PROTOCOL handles that match this controller.
  // This helps detect whether the firmware has published UsbIo handles (enumerated devices).
  //
  {
    UINTN HandleCount = 0;
    EFI_HANDLE *HandleBuffer = NULL;
    Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);
    if (!EFI_ERROR(Status) && HandleCount > 0) {
      Info(L"Total EFI_USB_IO_PROTOCOL handles in system: %d", HandleCount);
      // Inspect each USB_IO handle and compare device path (best-effort)
      for (UINTN i = 0; i < HandleCount; ++i) {
        EFI_USB_IO_PROTOCOL *UsbIo = NULL;
        Status = gBS->HandleProtocol(HandleBuffer[i], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
        if (EFI_ERROR(Status) || UsbIo == NULL) {
          continue;
        }
        // Safe device-level read: Device Descriptor
        EFI_USB_DEVICE_DESCRIPTOR DevDesc;
        Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
        if (!EFI_ERROR(Status)) {
          Info(L"  UsbIo handle[%d]: Vendor=0x%04x Product=0x%04x bDeviceClass=0x%02x bMaxPacketSize0=%d",
               i, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass, DevDesc.MaxPacketSize0);
        } else {
          Info(L"  UsbIo handle[%d]: UsbGetDeviceDescriptor failed (%r)", i, Status);
        }
      }
      if (HandleBuffer) {
        FreePool(HandleBuffer);
      }
    } else {
      Info(L"No EFI_USB_IO_PROTOCOL handles published (or LocateHandleBuffer failed: %r)", Status);
    }
  }

  Info(L"--- End inspect for this controller ---");
}

//
// === BEGIN: Added functions for EFI_USB_IO_PROTOCOL detection + raw-byte test ===
// (Paste these at the end of the existing file)
//

//
// === BEGIN: Added functions for EFI_USB_IO_PROTOCOL detection + raw-byte test ===
// (Paste these at the end of the existing file)
//
//
// Print buffer in hex
//

STATIC
VOID
PrintHexBuffer (
  IN UINT8 *Buf,
  IN UINTN Len
  )
{
  if (Buf == NULL || Len == 0) {
    Info(L"  <no data>");
    return;
  }

  // print bytes in lines of 16
  for (UINTN i = 0; i < Len; i += 16) {
    CHAR16 line[128];
    UINTN pos = 0;
    pos += UnicodeSPrint(line + pos, sizeof(line)/sizeof(CHAR16) - pos,
                        L"  %04x: ", (UINT32)i);
    for (UINTN j = 0; j < 16 && (i + j) < Len; ++j) {
      pos += UnicodeSPrint(line + pos, sizeof(line)/sizeof(CHAR16) - pos,
                           L"%02x ", Buf[i + j]);
    }
    Info(L"%s", line);
  }
}

STATIC
VOID
PrintHexLine (
  IN UINT8 *Buf,
  IN UINTN Len
  )
{
  if (Buf == NULL || Len == 0) {
    Info(L"    <no data>");
    return;
  }

  // print in a single line for small packets
  CHAR16 line[256];
  UINTN pos = 0;
  pos += UnicodeSPrint(line + pos, sizeof(line)/sizeof(CHAR16) - pos, L"    ");
  for (UINTN i = 0; i < Len; ++i) {
    pos += UnicodeSPrint(line + pos, sizeof(line)/sizeof(CHAR16) - pos, L"%02x ", Buf[i]);
    if (pos + 10 >= sizeof(line)/sizeof(CHAR16)) break;
  }
  Info(L"%s", line);
}

STATIC
VOID
ListenRawFirstHandle (
  IN UINTN Seconds
  )
{
  EFI_STATUS Status;
  UINTN HandleCount = 0;
  EFI_HANDLE *HandleBuffer = NULL;

  Info(L"[ListenRaw] Preparing to listen for %d seconds on first UsbIo handle", (UINT32)Seconds);

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Info(L"[ListenRaw] No EFI_USB_IO_PROTOCOL handles found (%r)", Status);
    if (HandleBuffer) FreePool(HandleBuffer);
    return;
  }

  // Use the first handle only (as requested)
  EFI_USB_IO_PROTOCOL *UsbIo = NULL;
  Status = gBS->HandleProtocol(HandleBuffer[0], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
  if (EFI_ERROR(Status) || UsbIo == NULL) {
    Info(L"[ListenRaw] Failed to open first UsbIo handle (%r)", Status);
    FreePool(HandleBuffer);
    return;
  }

  // Print a quick device descriptor attempt (best-effort)
  EFI_USB_DEVICE_DESCRIPTOR DevDesc;
  Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
  if (!EFI_ERROR(Status)) {
    Info(L"[ListenRaw] First handle: VID=0x%04x PID=0x%04x bDeviceClass=0x%02x",
         DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
  } else {
    Info(L"[ListenRaw] UsbGetDeviceDescriptor failed (%r)", Status);
  }

  // Search endpoints (1..16) for Interrupt IN first, else Bulk IN
  UINT8  FoundEpAddr = 0;
  UINT16 FoundMaxPacket = 0;
  UINT8  FoundType = 0; // 1=Interrupt, 2=Bulk

  for (UINT8 ep = 1; ep <= 16; ++ep) {
    EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
    Status = UsbIo->UsbGetEndpointDescriptor(UsbIo, ep, &EpDesc);
    if (EFI_ERROR(Status)) {
      continue;
    }
    UINT8 epAddr = EpDesc.EndpointAddress;
    UINT8 epType = EpDesc.Attributes & 0x03; // 0=ctrl,1=iso,2=bulk,3=interrupt
    UINT16 mps = EpDesc.MaxPacketSize;

    // Looking for IN endpoints (0x80 bit set)
    if ((epAddr & 0x80) == 0) continue;

    // Prefer interrupt IN
    if (epType == 0x03) {
      FoundEpAddr = epAddr;
      FoundMaxPacket = mps ? mps : 8;
      FoundType = 1;
      Info(L"[ListenRaw] Found Interrupt-IN endpoint 0x%02x MaxPacket=%d", epAddr, mps);
      break;
    }
    // Otherwise candidate bulk-in if not found yet
    if (epType == 0x02 && FoundEpAddr == 0) {
      FoundEpAddr = epAddr;
      FoundMaxPacket = mps ? mps : 64;
      FoundType = 2;
      Info(L"[ListenRaw] Found Bulk-IN endpoint 0x%02x MaxPacket=%d (using as fallback)", epAddr, mps);
      // don't break yet — keep searching in case an interrupt endpoint exists later
    }
  }

  if (FoundEpAddr == 0) {
    Info(L"[ListenRaw] No IN endpoints (Interrupt or Bulk) found on first handle — cannot listen.");
    FreePool(HandleBuffer);
    return;
  }

  // Defensive buffer size
  if (FoundMaxPacket == 0) {
    FoundMaxPacket = 64;
  }
  Info(L"[ListenRaw] Listening on endpoint 0x%02x for %d seconds (per-iteration timeout 1s).", FoundEpAddr, (UINT32)Seconds);

  // Loop Seconds times with 1-second blocking transfers (approx Seconds duration).
  for (UINTN iter = 0; iter < Seconds; ++iter) {
    UINTN DataLen = (UINTN)FoundMaxPacket;
    UINT8 *Buffer = AllocateZeroPool(DataLen);
    if (Buffer == NULL) {
      Info(L"[ListenRaw] Failed to allocate buffer of %d bytes", FoundMaxPacket);
      break;
    }
    UINT32 UsbStatus = 0;
    EFI_STATUS XferStatus;

    if (FoundType == 1) {
      // Interrupt IN
      XferStatus = UsbIo->UsbSyncInterruptTransfer(UsbIo, FoundEpAddr, Buffer, &DataLen, 1 /* timeout sec */, &UsbStatus);
    } else {
      // Bulk IN (fallback)
      XferStatus = UsbIo->UsbBulkTransfer(UsbIo, FoundEpAddr, Buffer, &DataLen, 1 /* timeout sec */, &UsbStatus);
    }

    if (!EFI_ERROR(XferStatus)) {
      if (DataLen > 0) {
        Info(L"[ListenRaw] Iter %d: received %d bytes (UsbStatus=%u)", (UINT32)iter, (UINT32)DataLen, UsbStatus);
        PrintHexLine(Buffer, DataLen);
      } else {
        Info(L"[ListenRaw] Iter %d: transfer completed but length==0 (UsbStatus=%u)", (UINT32)iter, UsbStatus);
      }
    } else {
      if (XferStatus == EFI_TIMEOUT) {
        Info(L"[ListenRaw] Iter %d: timeout (no data in this interval).", (UINT32)iter);
      } else {
        Info(L"[ListenRaw] Iter %d: transfer error %r (UsbStatus=%u)", (UINT32)iter, XferStatus, UsbStatus);
      }
    }

    FreePool(Buffer);
    // next iteration; each transfer used up to 1s timeout -> approx Seconds total
  }

  Info(L"[ListenRaw] Done listening on first handle endpoint 0x%02x", FoundEpAddr);

  FreePool(HandleBuffer);
}

//
// Try to identify device type from device/interface descriptor values.
//
STATIC
VOID
IdentifyDeviceClassFromUsbIo (
  IN EFI_USB_IO_PROTOCOL *UsbIo,
  IN UINTN               HandleIndex
  )
{
  EFI_STATUS Status;
  EFI_USB_DEVICE_DESCRIPTOR DevDesc;
  EFI_USB_INTERFACE_DESCRIPTOR IfDesc;
  CHAR16 *hint = L"(unknown)";

  // Device descriptor (may report device-level class)
  Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
  if (EFI_ERROR(Status)) {
    Info(L"UsbIo handle[%d]: UsbGetDeviceDescriptor failed (%r)", HandleIndex, Status);
  } else {
    Info(L"UsbIo handle[%d]: VID=0x%04x PID=0x%04x bDeviceClass=0x%02x bDeviceSubClass=0x%02x bDeviceProtocol=0x%02x",
         HandleIndex, DevDesc.IdVendor, DevDesc.IdProduct,
         DevDesc.DeviceClass, DevDesc.DeviceSubClass, DevDesc.DeviceProtocol);
  }

  // Interface descriptor (many devices use interface class to indicate HID)
  Status = UsbIo->UsbGetInterfaceDescriptor(UsbIo, &IfDesc);
  if (EFI_ERROR(Status)) {
    Info(L"UsbIo handle[%d]: UsbGetInterfaceDescriptor failed (%r)", HandleIndex, Status);
  } else {
    Info(L" UsbIf: bInterfaceClass=0x%02x bInterfaceSubClass=0x%02x bInterfaceProtocol=0x%02x",
         IfDesc.InterfaceClass, IfDesc.InterfaceSubClass, IfDesc.InterfaceProtocol);

    // Simple mapping for common classes
    switch (IfDesc.InterfaceClass) {
    case 0x03: // HID
      if (IfDesc.InterfaceProtocol == 0x01) {
        hint = L"HID keyboard (boot protocol)";
      } else if (IfDesc.InterfaceProtocol == 0x02) {
        hint = L"HID mouse (boot protocol)";
      } else {
        hint = L"HID (unknown subclass/protocol)";
      }
      break;
    case 0x08: // Mass Storage
      hint = L"Mass Storage (MSC)";
      break;
    case 0x0A: // CDC Data (network/serial)
      hint = L"CDC/ACM (serial/ethernet-like)";
      break;
    case 0x02: // Communications and CDC
      hint = L"Communications / CDC";
      break;
    case 0xFF:
      hint = L"Vendor-specific device";
      break;
    default:
      // some devices place class at device-level
      if (!EFI_ERROR(Status) && DevDesc.DeviceClass != 0) {
        switch (DevDesc.DeviceClass) {
        case 0x03:
          hint = L"HID (device-level)";
          break;
        case 0x08:
          hint = L"Mass Storage (device-level)";
          break;
        default:
          hint = L"(unknown class)";
        }
      }
      break;
    }
    Info(L"  -> Identification hint: %s", hint);
  }

  // Enumerate endpoint descriptors (print them, highlight INTERRUPT IN endpoints)
  for (UINT8 epIdx = 1; epIdx <= 16; ++epIdx) {
    EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
    Status = UsbIo->UsbGetEndpointDescriptor(UsbIo, epIdx, &EpDesc);
    if (EFI_ERROR(Status)) {
      // stop if not found (common)
      // Note: some implementations return EFI_NOT_FOUND when index too large
      continue;
    }
    UINT8 epAddr = EpDesc.EndpointAddress;
    UINT8 attrib = EpDesc.Attributes;
    UINT16 mps = EpDesc.MaxPacketSize;
    Info(L"  Endpoint[%d]: Addr=0x%02x Attr=0x%02x MaxPacket=%d Interval=%d",
         epIdx, epAddr, attrib, mps, EpDesc.Interval);

    if (((attrib & 0x03) == 0x03) && (epAddr & 0x80)) {
      Info(L"    -> Interrupt IN endpoint detected (likely HID/keyboard/mouse or status endpoint)");
    }
  }
}

//
// Find the first Interrupt IN endpoint for this UsbIo handle and return:
//   * endpoint address (0 if none found)
//   * pointer to max packet size (out param)
//
STATIC
UINT8
FindFirstInterruptInEndpoint (
  IN  EFI_USB_IO_PROTOCOL *UsbIo,
  OUT UINT16              *MaxPacketSizeOut
  )
{
  EFI_STATUS Status;
  if (MaxPacketSizeOut) {
    *MaxPacketSizeOut = 0;
  }

  for (UINT8 epIdx = 1; epIdx <= 16; ++epIdx) {
    EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
    Status = UsbIo->UsbGetEndpointDescriptor(UsbIo, epIdx, &EpDesc);
    if (EFI_ERROR(Status)) {
      continue;
    }
    // Attributes low 2 bits indicate transfer type: 0=Control,1=Iso,2=Bulk,3=Interrupt
    if (((EpDesc.Attributes & 0x03) == 0x03) && (EpDesc.EndpointAddress & 0x80)) {
      if (MaxPacketSizeOut) {
        *MaxPacketSizeOut = EpDesc.MaxPacketSize;
      }
      return EpDesc.EndpointAddress;
    }
  }
  return 0;
}

//
// Test: For each UsbIo handle that has an Interrupt IN endpoint, listen for raw bytes
// using UsbSyncInterruptTransfer for ~5 seconds and print raw packets as hex bytes.
// (This is the "real mode" raw bytes viewer the user requested.)
//
STATIC
VOID
TestListenRawForFiveSecondsPerDevice (
  VOID
  )
{
  EFI_STATUS Status;
  UINTN HandleCount = 0;
  EFI_HANDLE *HandleBuffer = NULL;

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Info(L"[RawTest] No EFI_USB_IO_PROTOCOL handles found (%r)", Status);
    if (HandleBuffer) {
      FreePool(HandleBuffer);
    }
    return;
  }

  Info(L"[RawTest] Found %d UsbIo handles. Starting raw listen (5s) on interrupt-IN endpoints.", HandleCount);

  // iterate all handles
  for (UINTN h = 0; h < HandleCount; ++h) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(HandleBuffer[h], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
    if (EFI_ERROR(Status) || UsbIo == NULL) {
      continue;
    }

    // Identify & print device info
    Info(L"[RawTest] Handle %d:", (UINT32)h);
    IdentifyDeviceClassFromUsbIo(UsbIo, h);

    // Find an interrupt IN endpoint
    UINT16 MaxPacket = 0;
    UINT8 EpAddr = FindFirstInterruptInEndpoint(UsbIo, &MaxPacket);
    if (EpAddr == 0) {
      Info(L"  No Interrupt-IN endpoint found for handle %d (skipping raw listen)", (UINT32)h);
      continue;
    }

    if (MaxPacket == 0) {
      // choose a reasonable default to avoid zero buffer
      MaxPacket = 8;
    }

    Info(L"  Listening on endpoint 0x%02x for ~5 seconds. Move device (e.g. mouse) now to see raw packets.", EpAddr);

    // We'll perform up to 5 synchronous interrupt transfers with 1 second timeout each,
    // which approximates a 5-second window but avoids blocking indefinitely.
    for (UINTN iter = 0; iter < 5; ++iter) {
      UINTN BufferLen = MaxPacket;
      UINT8 *Buffer = AllocateZeroPool(BufferLen);
      if (Buffer == NULL) {
        Info(L"  Failed to allocate buffer for listening");
        break;
      }
      UINT32 UsbStatus = 0;
      // Timeout is in SECONDS for UsbSyncInterruptTransfer (per header). Use 1 second.
      EFI_STATUS xfer = UsbIo->UsbSyncInterruptTransfer(UsbIo, EpAddr, Buffer, &BufferLen, 1, &UsbStatus);
      if (!EFI_ERROR(xfer)) {
        if (UsbStatus == EFI_USB_NOERROR && BufferLen > 0) {
          Info(L"  [Raw] Packet received (%d bytes):", (UINT32)BufferLen);
          PrintHexBuffer(Buffer, BufferLen);
        } else {
          Info(L"  [Raw] Transfer returned status=%u dataLen=%d (UsbStatus=%u)", xfer, (UINT32)BufferLen, UsbStatus);
        }
      } else {
        // timeout vs other error
        if (xfer == EFI_TIMEOUT) {
          Info(L"  [Raw] Timeout (no packet in this 1s interval).");
        } else {
          Info(L"  [Raw] UsbSyncInterruptTransfer returned %r (UsbStatus=%u)", xfer, UsbStatus);
        }
      }

      FreePool(Buffer);
    } // end per-handle 5s loop

    Info(L"  Finished listening on handle %d endpoint 0x%02x", (UINT32)h, EpAddr);
  }

  if (HandleBuffer) {
    FreePool(HandleBuffer);
  }
  Info(L"[RawTest] Completed.");
}

//
// Example wrapper you can call from UefiMain to run the test automatically.
// (Uncomment the call inside UefiMain if you want the test to run at the end
// of the existing checks.)
//
/*
  // Inside UefiMain, after your existing checks:
  TestListenRawForFiveSecondsPerDevice();
*/

//
// === END: Added functions ===
//



//
// DeepProbeUsbTopology - appended helper (read-only)
//  - Enumerates DevicePath handles and prints USB device-path nodes (USB, USB_CLASS, USB_WWID).
//  - Prints count & basic info for EFI_USB_IO_PROTOCOL handles.
//  - Scans PCI handles and looks for USB Host Controllers (class 0x0C / subclass 0x03)
//    and prints a conservative snapshot of port status registers (uses existing Pci/Mem helpers).
//
// Safe: read-only, uses LocateHandleBuffer + HandleProtocol and safe MemRead calls.
// Call this BEFORE your raw-listen test (it is designed to give deeper visibility).
//

STATIC
VOID
DeepProbeUsbTopology (
  VOID
  )
{
  EFI_STATUS Status;

  Info(L"\n--- DeepProbeUsbTopology: start ---");

  //
  // 1) How many EFI_USB_IO_PROTOCOL handles are published?
  //
  UINTN UsbIoCount = 0;
  EFI_HANDLE *UsbIoHandles = NULL;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbIoCount, &UsbIoHandles);
  if (EFI_ERROR(Status) || UsbIoCount == 0) {
    Info(L"[DeepProbe] EFI_USB_IO_PROTOCOL handles: 0 (LocateHandleBuffer returned %r)", Status);
    if (UsbIoHandles) FreePool(UsbIoHandles);
  } else {
    Info(L"[DeepProbe] EFI_USB_IO_PROTOCOL handles: %d", UsbIoCount);
    // Print brief descriptors for each published handle
    for (UINTN i = 0; i < UsbIoCount; ++i) {
      EFI_USB_IO_PROTOCOL *UsbIo = NULL;
      Status = gBS->HandleProtocol(UsbIoHandles[i], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
      if (!EFI_ERROR(Status) && UsbIo != NULL) {
        EFI_USB_DEVICE_DESCRIPTOR DevDesc;
        Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
        if (!EFI_ERROR(Status)) {
          Info(L"  UsbIo[%02d] VID=0x%04x PID=0x%04x bDeviceClass=0x%02x bNumConfigs=%d",
               (UINT32)i, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass, DevDesc.NumConfigurations);
        } else {
          Info(L"  UsbIo[%02d] : UsbGetDeviceDescriptor failed (%r)", (UINT32)i, Status);
        }
      } else {
        Info(L"  UsbIo[%02d] : HandleProtocol failed (%r)", (UINT32)i, Status);
      }
    }
    FreePool(UsbIoHandles);
  }

  //
  // 2) Walk DevicePath handles to find USB device-path nodes (MSG_USB_DP etc).
  //    This helps find devices that the firmware knows about even if UsbIo isn't published.
  //
  UINTN DpHandleCount = 0;
  EFI_HANDLE *DpHandles = NULL;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiDevicePathProtocolGuid, NULL, &DpHandleCount, &DpHandles);
  if (EFI_ERROR(Status) || DpHandleCount == 0) {
    Info(L"[DeepProbe] No DevicePath handles found (%r)", Status);
  } else {
    Info(L"[DeepProbe] Scanning %d DevicePath handles for USB nodes", DpHandleCount);

    for (UINTN h = 0; h < DpHandleCount; ++h) {
      EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
      Status = gBS->HandleProtocol(DpHandles[h], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
      if (EFI_ERROR(Status) || DevPath == NULL) {
        continue;
      }

      // Walk nodes
      EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
      BOOLEAN printedHandleHeader = FALSE;
      while (1) {
        UINT8 NodeType = Node->Type;
        UINT8 NodeSub = Node->SubType;
        UINT16 NodeLen = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));

        // End node?
        if (NodeType == 0x7F && NodeSub == 0xFF) {
          break;
        }

        // Only print header per-handle when we find the first USB node
        if (!printedHandleHeader) {
          Info(L"\n[DevPathHandle %p] scanning device-path nodes:", DpHandles[h]);
          printedHandleHeader = TRUE;
        }

        // Messaging device path and USB subtypes of interest:
        // MSG_USB_DP (0x05), MSG_USB_CLASS_DP (0x0f), MSG_USB_WWID_DP (0x10)
        if (NodeType == MESSAGING_DEVICE_PATH) {
          if (NodeSub == MSG_USB_DP) {
            // USB Device Path node: ParentPortNumber (UINT8), InterfaceNumber (UINT8)
            typedef struct {
              EFI_DEVICE_PATH_PROTOCOL Header;
              UINT8 ParentPortNumber;
              UINT8 InterfaceNumber;
            } _USB_DP;
            _USB_DP *U = (_USB_DP *)Node;
            Info(L"  MSG_USB_DP: ParentPort=%d Interface=%d", (UINT32)U->ParentPortNumber, (UINT32)U->InterfaceNumber);
          } else if (NodeSub == MSG_USB_CLASS_DP) {
            // USB class device path: VendorId(UINT16), ProductId(UINT16),
            // DeviceClass(UINT8), DeviceSubClass(UINT8), DeviceProtocol(UINT8)
            typedef struct {
              EFI_DEVICE_PATH_PROTOCOL Header;
              UINT16 VendorId;
              UINT16 ProductId;
              UINT8 DeviceClass;
              UINT8 DeviceSubClass;
              UINT8 DeviceProtocol;
            } _USB_CLASS_DP;
            _USB_CLASS_DP *C = (_USB_CLASS_DP *)Node;
            Info(L"  MSG_USB_CLASS_DP: VID=0x%04x PID=0x%04x Class=0x%02x Sub=0x%02x Prot=0x%02x",
                 C->VendorId, C->ProductId, C->DeviceClass, C->DeviceSubClass, C->DeviceProtocol);
          } else if (NodeSub == MSG_USB_WWID_DP) {
            // USB WWID DP: InterfaceNumber(UINT16), VendorId(UINT16), ProductId(UINT16), Serial unicode follows
            typedef struct {
              EFI_DEVICE_PATH_PROTOCOL Header;
              UINT16 InterfaceNumber;
              UINT16 VendorId;
              UINT16 ProductId;
              // CHAR16 Serial[] follows in node (variable)
            } _USB_WWID_DP;
            _USB_WWID_DP *W = (_USB_WWID_DP *)Node;
            // Compute pointer to serial chars if present
            CHAR16 *SerialStart = (CHAR16 *)(((UINT8*)Node) + sizeof(_USB_WWID_DP));
            UINTN SerialBytesLen = NodeLen - (UINT16)sizeof(EFI_DEVICE_PATH_PROTOCOL) - (UINT16)(sizeof(_USB_WWID_DP) - sizeof(EFI_DEVICE_PATH_PROTOCOL));
            // Serial length in chars (if positive)
            UINTN SerialChars = (SerialBytesLen > 1) ? (SerialBytesLen / 2) : 0;
            Info(L"  MSG_USB_WWID_DP: Interface=%d VID=0x%04x PID=0x%04x SerialLen=%d",
                 (UINT32)W->InterfaceNumber, W->VendorId, W->ProductId, (UINT32)SerialChars);
            if (SerialChars > 0) {
              // Print up to 64 chars defensively
              UINTN toPrint = SerialChars;
              if (toPrint > 64) toPrint = 64;
              CHAR16 tmp[65];
              CopyMem(tmp, SerialStart, toPrint * sizeof(CHAR16));
              tmp[toPrint] = L'\0';
              Info(L"    Serial (partial): %s", tmp);
            }
          }
        }

        // advance to next node
        if (NodeLen < sizeof(EFI_DEVICE_PATH_PROTOCOL)) break; // avoid infinite loop
        Node = (EFI_DEVICE_PATH_PROTOCOL *)(((UINT8*)Node) + NodeLen);
      } // end nodes
    } // end handles

    FreePool(DpHandles);
  }

  //
  // 3) Scan PCI handles and look for USB host controllers so we can probe port registers
  //    (this is similar to the inspection done earlier in your app but conservative)
  //
  UINTN PciCount = 0;
  EFI_HANDLE *PciHandles = NULL;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Info(L"[DeepProbe] No PCI handles found (%r)", Status);
  } else {
    Info(L"[DeepProbe] Scanning %d PCI handles for USB host controllers", PciCount);
    for (UINTN i = 0; i < PciCount; ++i) {
      EFI_PCI_IO_PROTOCOL *PciIo = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
      if (EFI_ERROR(Status) || PciIo == NULL) continue;

      UINT8 classCode = 0, subClass = 0, progIf = 0;
      ReadPci8(PciIo, PCI_CFG_OFFSET_CLASSCODE, &classCode);
      ReadPci8(PciIo, PCI_CFG_OFFSET_SUBCLASS, &subClass);
      ReadPci8(PciIo, PCI_CFG_OFFSET_PROGIF, &progIf);

      if (classCode == 0x0C && subClass == 0x03) {
        Info(L"  PCI USB HC found: ProgIf=0x%02x (handle %p)", progIf, PciHandles[i]);

        // Try reading CAPLENGTH at BAR + 0x0 (safe)
        UINT8 caplen = 0;
        if (!EFI_ERROR(MemRead8Block(PciIo, 0, 0x0, 1, &caplen))) {
          Info(L"    CapLength (BAR0+0x0) = 0x%02x", caplen);
        } else {
          Info(L"    Could not read cap length");
        }

        // Try to sample a small region (first 0x80) and print a couple of portSC values
        // We try offsets typical for EHCI/OHCI/xHCI but keep it defensive.
        UINT8 regs[0x80];
        if (!EFI_ERROR(MemRead8Block(PciIo, 0, 0x0, sizeof(regs), regs))) {
          // For EHCI-like, port SC base often at opbase + 0x44
          // We'll attempt to read opbase = caplen and then port 1 at opbase+0x44
          UINT64 opbase = (caplen ? caplen : 0);
          for (UINTN p = 1; p <= 8; ++p) {
            UINT64 off = opbase + EHCI_OP_PORTSC_BASE + 4 * (p - 1);
            UINT32 val = 0;
            if (!EFI_ERROR(MemRead32(PciIo, 0, off, &val))) {
              Info(L"    Port %d PORTSC @ op+0x%lx = 0x%08x", p, off - opbase, val);
            } else {
              // Skip printing repeated errors
            }
          }
        } else {
          Info(L"    Couldn't read a small register block at BAR0 (reads may be restricted)");
        }
      } // end if HC
    } // end pci handles
    FreePool(PciHandles);
  }

  Info(L"--- DeepProbeUsbTopology: end ---\n");
}

//
// How to use:
//   Call DeepProbeUsbTopology(); before calling ListenRawFirstHandle(5);
//   e.g. in UefiMain (after your Inspect loop) add:
//      Info(L\"UsbDiagApp: deep probe before RAW tests\"); DeepProbeUsbTopology();
//

//
// MonitorUsbTopologyForFiveSeconds()
//  - prints UsbIo handle count and device-path USB nodes (brief)
//  - polls all PCI USB host controllers' port status registers for ~5 seconds
//  - prints changes (connect/disconnect, PortOwner changes, enabled/reset bits)
//  - safe/read-only; no writes or resets
//
STATIC
VOID
MonitorUsbTopologyForFiveSeconds (
  VOID
  )
{
  EFI_STATUS Status;
  Info(L"\n--- MonitorUsbTopologyForFiveSeconds: start ---");

  // 1) Print UsbIo handle count
  UINTN UsbIoCount = 0;
  EFI_HANDLE *UsbIoHandles = NULL;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbIoCount, &UsbIoHandles);
  if (EFI_ERROR(Status) || UsbIoCount == 0) {
    Info(L"[Monitor] EFI_USB_IO_PROTOCOL handles: 0 (%r)", Status);
  } else {
    Info(L"[Monitor] EFI_USB_IO_PROTOCOL handles: %d", UsbIoCount);
    for (UINTN i = 0; i < UsbIoCount; ++i) {
      EFI_USB_IO_PROTOCOL *UsbIo = NULL;
      Status = gBS->HandleProtocol(UsbIoHandles[i], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
      if (!EFI_ERROR(Status) && UsbIo != NULL) {
        EFI_USB_DEVICE_DESCRIPTOR D;
        if (!EFI_ERROR(UsbIo->UsbGetDeviceDescriptor(UsbIo, &D))) {
          Info(L"  UsbIo[%02d] VID=0x%04x PID=0x%04x Class=0x%02x", (UINT32)i, D.IdVendor, D.IdProduct, D.DeviceClass);
        } else {
          Info(L"  UsbIo[%02d] UsbGetDeviceDescriptor failed", (UINT32)i);
        }
      }
    }
    FreePool(UsbIoHandles);
  }

  // 2) Print DevicePath USB nodes and record ParentPort numbers (help correlate physical ports)
  UINTN DpCount = 0;
  EFI_HANDLE *DpHandles = NULL;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiDevicePathProtocolGuid, NULL, &DpCount, &DpHandles);
  if (EFI_ERROR(Status) || DpCount == 0) {
    Info(L"[Monitor] DevicePath handles: 0 (%r)", Status);
  } else {
    Info(L"[Monitor] Scanning %d DevicePath handles for USB nodes", DpCount);
    for (UINTN h = 0; h < DpCount; ++h) {
      EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
      Status = gBS->HandleProtocol(DpHandles[h], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
      if (EFI_ERROR(Status) || DevPath == NULL) continue;
      EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
      BOOLEAN printed = FALSE;
      while (1) {
        if (Node->Type == 0x7F && Node->SubType == 0xFF) break;
        if (Node->Type == MESSAGING_DEVICE_PATH) {
          if (Node->SubType == MSG_USB_DP) {
            if (!printed) { Info(L"[DevPathHandle %p] USB nodes:", DpHandles[h]); printed = TRUE; }
            typedef struct { EFI_DEVICE_PATH_PROTOCOL H; UINT8 ParentPortNumber; UINT8 InterfaceNumber; } _U;
            _U *u = (_U*)Node;
            Info(L"  MSG_USB_DP: ParentPort=%d Interface=%d", (UINT32)u->ParentPortNumber, (UINT32)u->InterfaceNumber);
          } else if (Node->SubType == MSG_USB_CLASS_DP) {
            if (!printed) { Info(L"[DevPathHandle %p] USB nodes:", DpHandles[h]); printed = TRUE; }
            typedef struct { EFI_DEVICE_PATH_PROTOCOL H; UINT16 VendorId; UINT16 ProductId; UINT8 DeviceClass; UINT8 DeviceSubClass; UINT8 DeviceProtocol; } _C;
            _C *c = (_C*)Node;
            Info(L"  MSG_USB_CLASS_DP: VID=0x%04x PID=0x%04x Class=0x%02x", c->VendorId, c->ProductId, c->DeviceClass);
          } else if (Node->SubType == MSG_USB_WWID_DP) {
            if (!printed) { Info(L"[DevPathHandle %p] USB nodes:", DpHandles[h]); printed = TRUE; }
            typedef struct { EFI_DEVICE_PATH_PROTOCOL H; UINT16 InterfaceNumber; UINT16 VendorId; UINT16 ProductId; } _W;
            _W *w = (_W*)Node;
            Info(L"  MSG_USB_WWID_DP: Interface=%d VID=0x%04x PID=0x%04x", (UINT32)w->InterfaceNumber, w->VendorId, w->ProductId);
          }
        }
        // advance safely
        UINT16 len = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));
        if (len < sizeof(EFI_DEVICE_PATH_PROTOCOL)) break;
        Node = (EFI_DEVICE_PATH_PROTOCOL*)(((UINT8*)Node) + len);
      }
    }
    FreePool(DpHandles);
  }

  // 3) Find PCI handles for USB host controllers and snapshot their port states
  UINTN PciCount = 0;
  EFI_HANDLE *PciHandles = NULL;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Info(L"[Monitor] No PCI handles (%r)", Status);
    Info(L"--- MonitorUsbTopologyForFiveSeconds: end ---\n");
    return;
  }

  // Collect HC entries: handle + progIf + caplen + guess max ports
  typedef struct {
    EFI_HANDLE Handle;
    EFI_PCI_IO_PROTOCOL *PciIo;
    UINT8 ProgIf;
    UINT8 CapLen;
    UINTN MaxProbePorts;
  } HC_ENTRY;

  HC_ENTRY *Hubs = AllocateZeroPool(PciCount * sizeof(HC_ENTRY));
  UINTN HubCount = 0;

  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) continue;
    UINT8 classCode=0, subClass=0, progIf=0;
    ReadPci8(PciIo, PCI_CFG_OFFSET_CLASSCODE, &classCode);
    ReadPci8(PciIo, PCI_CFG_OFFSET_SUBCLASS, &subClass);
    ReadPci8(PciIo, PCI_CFG_OFFSET_PROGIF, &progIf);
    if (classCode == 0x0C && subClass == 0x03) {
      // Found HC
      Hubs[HubCount].Handle = PciHandles[i];
      Hubs[HubCount].PciIo  = PciIo;
      Hubs[HubCount].ProgIf = progIf;
      // read caplen safe
      UINT8 cap = 0;
      if (!EFI_ERROR(MemRead8Block(PciIo, 0, 0x0, 1, &cap))) { Hubs[HubCount].CapLen = cap; } else { Hubs[HubCount].CapLen = 0; }
      // choose probe width: for EHCI probe first 8, for others probe 8 as safe
      Hubs[HubCount].MaxProbePorts = 8;
      Info(L"[Monitor] Found PCI USB HC ProgIf=0x%02x handle=%p caplen=0x%02x", progIf, PciHandles[i], Hubs[HubCount].CapLen);
      HubCount++;
    }
  }

  if (PciHandles) FreePool(PciHandles);

  // If no hubs found, exit
  if (HubCount == 0) {
    Info(L"[Monitor] No USB host controllers found.");
    if (Hubs) FreePool(Hubs);
    Info(L"--- MonitorUsbTopologyForFiveSeconds: end ---\n");
    return;
  }

  // Prepare previous values array
  UINT32 *PrevVals = AllocateZeroPool(HubCount * 16 * sizeof(UINT32)); // support up to 16 ports per hub safe
  SetMem(PrevVals, HubCount * 16 * sizeof(UINT32), 0x00);

  // Initial snapshot and print
  for (UINTN h = 0; h < HubCount; ++h) {
    Info(L"[Monitor] Initial snapshot for HC %d (ProgIf=0x%02x):", (UINT32)h, Hubs[h].ProgIf);
    UINT64 opbase = (UINT64)(Hubs[h].CapLen ? Hubs[h].CapLen : 0);
    for (UINTN p = 1; p <= Hubs[h].MaxProbePorts; ++p) {
      UINT64 off = opbase + EHCI_OP_PORTSC_BASE + 4 * (p - 1);
      UINT32 val = 0;
      if (!EFI_ERROR(MemRead32(Hubs[h].PciIo, 0, off, &val))) {
        PrevVals[h * 16 + (p-1)] = val;
        Info(L"  Port %02d: 0x%08x", (UINT32)p, val);
      } else {
        PrevVals[h * 16 + (p-1)] = 0;
      }
    }
  }

  // Poll loop: ~5 seconds, sample every 500ms (10 iterations)
  CONST UINTN Iterations = 10;
  for (UINTN iter = 0; iter < Iterations; ++iter) {
    // Stall for 500ms
    gBS->Stall(500 * 1000); // microseconds

    for (UINTN h = 0; h < HubCount; ++h) {
      UINT64 opbase = (UINT64)(Hubs[h].CapLen ? Hubs[h].CapLen : 0);
      for (UINTN p = 1; p <= Hubs[h].MaxProbePorts; ++p) {
        UINT64 off = opbase + EHCI_OP_PORTSC_BASE + 4*(p-1);
        UINT32 val = 0;
        if (EFI_ERROR(MemRead32(Hubs[h].PciIo, 0, off, &val))) continue;
        UINT32 prev = PrevVals[h * 16 + (p-1)];
        if (val != prev) {
          Info(L"[Monitor] HC %d Port %d changed: 0x%08x -> 0x%08x", (UINT32)h, (UINT32)p, prev, val);
          // Interpret a few useful bits
          Info(L"    CCS=%d PE=%d PR=%d PO=%d SpeedBits=0x%02x",
               (val >> 0) & 1,
               (val >> 2) & 1,
               (val >> 8) & 1,
               (val >> 13) & 1,
               (val >> 10) & 0x7);
          PrevVals[h * 16 + (p-1)] = val;
        } // else unchanged
      }
    }
  }

  // Cleanup
  if (Hubs) FreePool(Hubs);
  if (PrevVals) FreePool(PrevVals);

  Info(L"--- MonitorUsbTopologyForFiveSeconds: end ---\n");
}

/////////////////////////////////////////////// ASYNC PART //////////////////////////////////////////////////
//
// AsyncMultiRawListener.c  -- appended functions (standalone)
//
// Adds:
//  - AsyncUsbCallback()              : callback invoked when an async interrupt packet arrives
//  - StartAsyncMultiHandleListen()   : start async listeners for all UsbIo handles with Interrupt IN endpoints
//  - StopAsyncMultiHandleListen()    : cancels active async transfers
//  - TryExtraInvestigations()        : small follow-up checks when no HID endpoints are found
//
// Usage: call StartAsyncMultiHandleListen(DurationSeconds) from UefiMain
//

typedef struct {
  EFI_USB_IO_PROTOCOL   *UsbIo;
  UINT8                 EpAddr;
  UINTN                 PollInterval;
  UINTN                 DataLength;
  UINTN                 HandleIndex;
  BOOLEAN               StopRequested;
} ASYNC_LISTENER_CTX;

//
// small hex printer (self-contained)
//
STATIC
VOID
PrintHexInline(
  IN VOID   *Buf,
  IN UINTN  Len
  )
{
  if (Buf == NULL || Len == 0) {
    Info(L"    <no data>");
    return;
  }

  UINT8 *b = (UINT8*)Buf;
  // print up to 64 bytes inline
  UINTN toPrint = Len;
  if (toPrint > 64) toPrint = 64;

  CHAR16 line[256];
  UINTN pos = 0;
  pos += UnicodeSPrint(line + pos, sizeof(line)/sizeof(CHAR16) - pos, L"    ");
  for (UINTN i = 0; i < toPrint; ++i) {
    pos += UnicodeSPrint(line + pos, sizeof(line)/sizeof(CHAR16) - pos, L"%02x ", b[i]);
    if (pos + 16 >= sizeof(line)/sizeof(CHAR16)) break;
  }
  if (toPrint < Len) {
    pos += UnicodeSPrint(line + pos, sizeof(line)/sizeof(CHAR16) - pos, L"... (%d bytes total)", (UINT32)Len);
  }
  Info(L"%s", line);
}

//
// Async callback called by UEFI when an async interrupt transfer completes.
// Matches EFI_ASYNC_USB_TRANSFER_CALLBACK:
//   EFI_STATUS (EFIAPI *)(VOID *Data, UINTN DataLength, VOID *Context, UINT32 Status)
//
STATIC
EFI_STATUS
EFIAPI
AsyncUsbCallback (
  IN VOID   *Data,
  IN UINTN  DataLength,
  IN VOID   *Context,
  IN UINT32 Status
  )
{
  ASYNC_LISTENER_CTX *Ctxt = (ASYNC_LISTENER_CTX*)Context;

  // Defensive: null-check
  if (Ctxt == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  // Print a short header for this arrival
  Info(L"[AsyncCB] Handle %d Ep 0x%02x Status=%u Received=%d bytes",
       (UINT32)Ctxt->HandleIndex, (UINT32)Ctxt->EpAddr, Status, (UINT32)DataLength);

  // Print raw bytes (truncated if long)
  PrintHexInline(Data, DataLength);

  // If StopRequested is not set, re-submit another async transfer so we keep listening.
  // Per UEFI spec: to start a new async transfer call with IsNewTransfer == TRUE.
  if (!Ctxt->StopRequested) {
    EFI_STATUS S = Ctxt->UsbIo->UsbAsyncInterruptTransfer(
                      Ctxt->UsbIo,
                      Ctxt->EpAddr,
                      TRUE,               // IsNewTransfer == start
                      Ctxt->PollInterval, // PollingInterval optional (units: frames/intervals; common small value)
                      Ctxt->DataLength,   // DataLength (size of buffer requested)
                      AsyncUsbCallback,   // Callback
                      Ctxt                // Context
                    );
    if (EFI_ERROR(S)) {
      Info(L"[AsyncCB] Failed to re-submit async transfer for handle %d ep 0x%02x (%r)", (UINT32)Ctxt->HandleIndex, (UINT32)Ctxt->EpAddr, S);
    }
  } else {
    Info(L"[AsyncCB] Stop requested, not re-submitting for handle %d", (UINT32)Ctxt->HandleIndex);
  }

  return EFI_SUCCESS;
}

//
// Start asynchronous listeners for all published EFI_USB_IO_PROTOCOL handles
// that expose an Interrupt-IN endpoint. Each device gets its own context; when
// a packet arrives the AsyncUsbCallback will print it and re-submit a new transfer.
//
// DurationSeconds: how many seconds to listen (main thread will block while callbacks run).
// Returns number of listeners started (0 possible).
//
STATIC
UINTN
StartAsyncMultiHandleListen (
  IN UINTN DurationSeconds
  )
{
  EFI_STATUS Status;
  UINTN HandleCount = 0;
  EFI_HANDLE *HandleBuffer = NULL;
  UINTN ActiveListeners = 0;

  Info(L"\n[AsyncListen] Starting multi-handle async listener for %d second(s)", (UINT32)DurationSeconds);

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Info(L"[AsyncListen] No EFI_USB_IO_PROTOCOL handles (%r)", Status);
    if (HandleBuffer) FreePool(HandleBuffer);
    return 0;
  }

  // Allocate contexts array
  ASYNC_LISTENER_CTX **CtxArray = AllocateZeroPool(sizeof(ASYNC_LISTENER_CTX*) * HandleCount);
  if (CtxArray == NULL) {
    Info(L"[AsyncListen] Allocation failure");
    FreePool(HandleBuffer);
    return 0;
  }

  // For each handle, try to find an Interrupt-IN endpoint
  for (UINTN h = 0; h < HandleCount; ++h) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(HandleBuffer[h], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
    if (EFI_ERROR(Status) || UsbIo == NULL) {
      continue;
    }

    // Optional: print device descriptor brief
    EFI_USB_DEVICE_DESCRIPTOR DevDesc;
    if (!EFI_ERROR(UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc))) {
      Info(L"[AsyncListen] Handle %d: VID=0x%04x PID=0x%04x bDeviceClass=0x%02x",
           (UINT32)h, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
    }

    // Look for INTERRUPT IN endpoint up to 16 endpoints
    UINT8 FoundEp = 0;
    UINTN FoundMps = 0;
    for (UINT8 epIdx = 1; epIdx <= 16; ++epIdx) {
      EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
      Status = UsbIo->UsbGetEndpointDescriptor(UsbIo, epIdx, &EpDesc);
      if (EFI_ERROR(Status)) {
        continue;
      }
      // Attributes low 2 bits indicate transfer type: 3 == interrupt
      UINT8 type = EpDesc.Attributes & 0x03;
      UINT8 addr = EpDesc.EndpointAddress;
      if (type == 0x03 && (addr & 0x80)) {
        FoundEp = addr;
        FoundMps = EpDesc.MaxPacketSize ? EpDesc.MaxPacketSize : 8;
        Info(L"[AsyncListen] Handle %d: Interrupt-IN endpoint found Ep=0x%02x MaxPacket=%d Interval=%d",
             (UINT32)h, FoundEp, (UINT32)FoundMps, EpDesc.Interval);
        break;
      }
    }

    if (FoundEp == 0) {
      Info(L"[AsyncListen] Handle %d: no Interrupt-IN endpoint (skip)", (UINT32)h);
      CtxArray[h] = NULL;
      continue;
    }

    // Build a context and start async transfer
    ASYNC_LISTENER_CTX *C = AllocateZeroPool(sizeof(ASYNC_LISTENER_CTX));
    if (C == NULL) {
      Info(L"[AsyncListen] Failed to allocate context for handle %d", (UINT32)h);
      CtxArray[h] = NULL;
      continue;
    }
    C->UsbIo = UsbIo;
    C->EpAddr = FoundEp;
    C->PollInterval = 10;        // reasonable polling interval; small. (UEFI interpretation varies; 10 is typical)
    C->DataLength = FoundMps;    // request MPS sized buffer
    C->HandleIndex = h;
    C->StopRequested = FALSE;

    // Start async transfer (IsNewTransfer == TRUE)
    Status = C->UsbIo->UsbAsyncInterruptTransfer(
                C->UsbIo,
                C->EpAddr,
                TRUE,               // start new async transfer
                C->PollInterval,    // polling interval
                C->DataLength,      // buffer length to request by underlying stack
                AsyncUsbCallback,   // callback invoked when data arrives
                C                   // context passed into callback
             );
    if (EFI_ERROR(Status)) {
      Info(L"[AsyncListen] UsbAsyncInterruptTransfer failed for handle %d ep 0x%02x (%r)", (UINT32)h, (UINT32)C->EpAddr, Status);
      FreePool(C);
      CtxArray[h] = NULL;
      continue;
    }

    // store the context
    CtxArray[h] = C;
    ActiveListeners++;
  } // end handle loop

  if (ActiveListeners == 0) {
    Info(L"[AsyncListen] No interrupt-IN endpoints found on any published UsbIo handles. Nothing to listen on.");
    // free contexts array and handle buffer
    FreePool(CtxArray);
    FreePool(HandleBuffer);
    return 0;
  }

  //
  // Wait DurationSeconds while callbacks are delivered asynchronously.
  // Use a single timer event to block this thread; callbacks fire on transfers.
  //
  EFI_EVENT Timer = NULL;
  Status = gBS->CreateEvent(EVT_TIMER, TPL_CALLBACK, NULL, NULL, &Timer);
  if (!EFI_ERROR(Status)) {
    // Set relative timer: DurationSeconds * 10M (100ns units)
    UINT64 Rel = (UINT64)DurationSeconds * 10000000ULL; // 1s = 10,000,000 * 100ns
    gBS->SetTimer(Timer, TimerRelative, Rel);

    // Wait for timer to signal (callbacks still executed by the environment)
    UINTN Index = 0;
    gBS->WaitForEvent(1, &Timer, &Index);

    // Destroy timer event
    gBS->CloseEvent(Timer);
  } else {
    // If timer creation failed, fallback to repeated stalls
    for (UINTN s = 0; s < DurationSeconds; ++s) {
      gBS->Stall(1000 * 1000); // 1 second
    }
  }

  //
  // Timer expired -> request stop and cancel async transfers.
  //
  Info(L"[AsyncListen] Duration elapsed. Cancelling %d active listeners...", (UINT32)ActiveListeners);

  for (UINTN i = 0; i < HandleCount; ++i) {
    ASYNC_LISTENER_CTX *C = CtxArray[i];
    if (C == NULL) continue;

    // Request callback not to re-submit
    C->StopRequested = TRUE;

    // Per UEFI spec: calling UsbAsyncInterruptTransfer with IsNewTransfer == FALSE cancels the transfer on that endpoint
    EFI_STATUS S = C->UsbIo->UsbAsyncInterruptTransfer(C->UsbIo, C->EpAddr, FALSE, 0, 0, NULL, NULL);
    if (EFI_ERROR(S)) {
      Info(L"[AsyncListen] Cancel request for handle %d ep 0x%02x returned %r", (UINT32)C->HandleIndex, (UINT32)C->EpAddr, S);
    } else {
      Info(L"[AsyncListen] Cancel requested for handle %d ep 0x%02x", (UINT32)C->HandleIndex, (UINT32)C->EpAddr);
    }

    // free context (callbacks may still run one last time depending on implementation timing,
    // but StopRequested is set so callback won't requeue).
    FreePool(C);
    CtxArray[i] = NULL;
  }

  FreePool(CtxArray);
  FreePool(HandleBuffer);

  Info(L"[AsyncListen] Finished cancelling listeners.");
  return ActiveListeners;
}

//
// Optional: if no interrupt endpoints were found, do a few extra investigations
//  - re-print DevicePath USB nodes (already done elsewhere)
//  - print a small suggestive message about PortOwner/companion controllers
//
STATIC
VOID
TryExtraInvestigationsAfterAsyncListen (
  VOID
  )
{
  Info(L"\n[Investigate] Extra checks after Async listen:");

  // Re-run DevicePath scan summary (short), to show any device-path USB nodes
  UINTN DpCount = 0;
  EFI_HANDLE *DpHandles = NULL;
  EFI_STATUS Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiDevicePathProtocolGuid, NULL, &DpCount, &DpHandles);
  if (EFI_ERROR(Status) || DpCount == 0) {
    Info(L"[Investigate] No DevicePath handles: %r", Status);
  } else {
    Info(L"[Investigate] DevicePath handles: %d (looking for MSG_USB_DP)", DpCount);
    for (UINTN h = 0; h < DpCount; ++h) {
      EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
      Status = gBS->HandleProtocol(DpHandles[h], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
      if (EFI_ERROR(Status) || DevPath == NULL) continue;
      EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
      while (1) {
        if (Node->Type == 0x7F && Node->SubType == 0xFF) break;
        if (Node->Type == MESSAGING_DEVICE_PATH && Node->SubType == MSG_USB_DP) {
          typedef struct { EFI_DEVICE_PATH_PROTOCOL H; UINT8 ParentPortNumber; UINT8 InterfaceNumber; } _U;
          _U *u = (_U*)Node;
          Info(L"  DevPath %p: MSG_USB_DP ParentPort=%d Interface=%d", DpHandles[h], (UINT32)u->ParentPortNumber, (UINT32)u->InterfaceNumber);
        }
        UINT16 len = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));
        if (len < sizeof(EFI_DEVICE_PATH_PROTOCOL)) break;
        Node = (EFI_DEVICE_PATH_PROTOCOL*)(((UINT8*)Node) + len);
      }
    }
    FreePool(DpHandles);
  }

  // Print suggestion: PortOwner bits and companion controllers commonly cause devices to be handled
  Info(L"[Investigate] Tip: many ports can be owned by companion controllers (EHCI/OHCI/UHCI split).");
  Info(L"[Investigate] If PortOwner==1 on EHCI ports, those ports are controlled by companion HC and may show up under other HC or OS drivers.");
  Info(L"[Investigate] Use MonitorUsbTopologyForFiveSeconds to sample PORTSC registers and look for CCS/PE/PR/PO bits (we already do this earlier).");
}

//
// Convenience wrapper for UefiMain: run async multi-listen for 5 seconds then investigate.
// Call this from your UefiMain at the time you want to run async listening.
// Example: StartAsyncMultiHandleListen(5); TryExtraInvestigationsAfterAsyncListen();
//

//
// ----- Safe probes: HID report descriptor and Hub descriptor readers -----
//
// Usage: call FetchHidReportDescriptors(); and FetchHubDescriptors(); from UefiMain
//

//
// Fixed: safe calls to UsbControlTransfer (cast to known function-pointer type)
// and safe parsing of hub wHubChar (no unaligned UINT16 dereference).
//

// typedef matching the common UEFI UsbControlTransfer signature we will use
typedef
EFI_STATUS
(EFIAPI *EFI_USB_CONTROL_TRANSFER_T) (
  IN EFI_USB_IO_PROTOCOL          *This,
  IN EFI_USB_DEVICE_REQUEST       *Request,
  IN EFI_USB_DATA_DIRECTION       Direction,
  IN UINTN                        Timeout,
  IN OUT VOID                     *Data,
  IN OUT UINTN                    *DataLength,
  OUT UINT32                      *UsbStatus
  );

STATIC
VOID
PrintHexInlineSmall(
  IN VOID   *Buf,
  IN UINTN  Len
  )
{
  if (Buf == NULL || Len == 0) {
    Info(L"    <no data>");
    return;
  }
  UINT8 *b = (UINT8*)Buf;
  UINTN toPrint = Len;
  if (toPrint > 256) toPrint = 256;
  CHAR16 tmp[512];
  UINTN pos = 0;
  pos += UnicodeSPrint(tmp + pos, sizeof(tmp)/sizeof(CHAR16) - pos, L"    ");
  for (UINTN i = 0; i < toPrint; ++i) {
    pos += UnicodeSPrint(tmp + pos, sizeof(tmp)/sizeof(CHAR16) - pos, L"%02x ", b[i]);
    if (pos + 32 >= sizeof(tmp)/sizeof(CHAR16)) break;
  }
  if (toPrint < Len) {
    pos += UnicodeSPrint(tmp + pos, sizeof(tmp)/sizeof(CHAR16) - pos, L"... (%d bytes)", (UINT32)Len);
  }
  Info(L"%s", tmp);
}

STATIC
VOID
FetchHidReportDescriptors (
  VOID
  )
{
  EFI_STATUS Status;
  UINTN HandleCount = 0;
  EFI_HANDLE *Handles = NULL;

  Info(L"\n--- FetchHidReportDescriptors: start ---");

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Info(L"[HIDProbe] No UsbIo handles (%r)", Status);
    if (Handles) FreePool(Handles);
    return;
  }

  for (UINTN h = 0; h < HandleCount; ++h) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(Handles[h], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
    if (EFI_ERROR(Status) || UsbIo == NULL) continue;

    EFI_USB_DEVICE_DESCRIPTOR DevDesc;
    if (!EFI_ERROR(UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc))) {
      Info(L"[HIDProbe] Handle %d VID=0x%04x PID=0x%04x bNumConfigs=%d", (UINT32)h, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.NumConfigurations);
    }

    EFI_USB_INTERFACE_DESCRIPTOR IfDesc;
    Status = UsbIo->UsbGetInterfaceDescriptor(UsbIo, &IfDesc);
    if (EFI_ERROR(Status)) {
      Info(L"[HIDProbe] Handle %d: UsbGetInterfaceDescriptor failed (%r) - skipping", (UINT32)h, Status);
      continue;
    }

    Info(L"[HIDProbe] Handle %d: InterfaceClass=0x%02x SubClass=0x%02x Protocol=0x%02x NumEndpoints=%d InterfaceNumber=%d",
         (UINT32)h, IfDesc.InterfaceClass, IfDesc.InterfaceSubClass, IfDesc.InterfaceProtocol, IfDesc.NumEndpoints, IfDesc.InterfaceNumber);

    if (IfDesc.InterfaceClass != 0x03) {
      Info(L"[HIDProbe] Handle %d: Not HID class - skipping HID report fetch", (UINT32)h);
      continue;
    }

    // Prepare GET_DESCRIPTOR (Report 0x22) request
    EFI_USB_DEVICE_REQUEST DevReq;
    ZeroMem(&DevReq, sizeof(DevReq));
    DevReq.RequestType = 0xA1;          // IN, Class, Interface
    DevReq.Request     = 0x06;          // GET_DESCRIPTOR
    DevReq.Value       = (0x22 << 8);   // Report descriptor type
    DevReq.Index       = (UINT16)IfDesc.InterfaceNumber;
    DevReq.Length      = 256;           // request up to 256 bytes

    UINT32 UsbStatus = 0;
    UINTN DataLen = (UINTN)DevReq.Length;
    VOID *Buf = AllocateZeroPool(DataLen);
    if (Buf == NULL) {
      Info(L"[HIDProbe] Allocation failed");
      continue;
    }

    // Call through a well-typed function pointer to avoid prototype mismatch warnings
    EFI_USB_CONTROL_TRANSFER_T Ctrl = (EFI_USB_CONTROL_TRANSFER_T)UsbIo->UsbControlTransfer;
    Status = Ctrl(UsbIo, &DevReq, EfiUsbDataIn, 5000, Buf, &DataLen, &UsbStatus);

    if (!EFI_ERROR(Status) && DataLen > 0) {
      Info(L"[HIDProbe] Handle %d: HID Report descriptor %d bytes received (UsbStatus=%u):", (UINT32)h, (UINT32)DataLen, UsbStatus);
      PrintHexInlineSmall(Buf, DataLen);
    } else {
      Info(L"[HIDProbe] Handle %d: Failed to get HID Report descriptor (%r) UsbStatus=%u", (UINT32)h, Status, UsbStatus);
    }

    FreePool(Buf);
  }

  FreePool(Handles);
  Info(L"--- FetchHidReportDescriptors: end ---\n");
}

STATIC
VOID
FetchHubDescriptors (
  VOID
  )
{
  EFI_STATUS Status;
  UINTN HandleCount = 0;
  EFI_HANDLE *Handles = NULL;

  Info(L"\n--- FetchHubDescriptors: start ---");

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Info(L"[HubProbe] No UsbIo handles (%r)", Status);
    if (Handles) FreePool(Handles);
    return;
  }

  for (UINTN h = 0; h < HandleCount; ++h) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(Handles[h], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
    if (EFI_ERROR(Status) || UsbIo == NULL) continue;

    EFI_USB_DEVICE_DESCRIPTOR DevDesc;
    if (!EFI_ERROR(UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc))) {
      Info(L"[HubProbe] Handle %d VID=0x%04x PID=0x%04x bDeviceClass=0x%02x", (UINT32)h, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
    }

    // Check for hub class at device or interface level
    BOOLEAN IsHub = FALSE;
    if (DevDesc.DeviceClass == 0x09) {
      IsHub = TRUE;
    } else {
      EFI_USB_INTERFACE_DESCRIPTOR IfDesc;
      if (!EFI_ERROR(UsbIo->UsbGetInterfaceDescriptor(UsbIo, &IfDesc))) {
        if (IfDesc.InterfaceClass == 0x09) {
          IsHub = TRUE;
        }
      }
    }
    if (!IsHub) {
      Info(L"[HubProbe] Handle %d: not hub (device/interface class != 0x09) - skipping", (UINT32)h);
      continue;
    }

    Info(L"[HubProbe] Handle %d appears to be a hub. Attempting GET_DESCRIPTOR for hub types", (UINT32)h);

    UINT8 hubTypes[] = { 0x29, 0x2A }; // USB2 Hub, SuperSpeed Hub
    for (UINTN t = 0; t < sizeof(hubTypes)/sizeof(hubTypes[0]); ++t) {
      UINT8 dtype = hubTypes[t];

      EFI_USB_DEVICE_REQUEST DevReq;
      ZeroMem(&DevReq, sizeof(DevReq));
      DevReq.RequestType = 0xA0;       // IN, Class, Device (hub class)
      DevReq.Request     = 0x06;       // GET_DESCRIPTOR
      DevReq.Value       = (UINT16)(dtype << 8);
      DevReq.Index       = 0;
      DevReq.Length      = 256;

      UINTN DataLen = (UINTN)DevReq.Length;
      VOID *Buf = AllocateZeroPool(DataLen);
      if (Buf == NULL) {
        Info(L"[HubProbe] Allocation failed");
        continue;
      }
      UINT32 UsbStatus = 0;

      EFI_USB_CONTROL_TRANSFER_T Ctrl = (EFI_USB_CONTROL_TRANSFER_T)UsbIo->UsbControlTransfer;
      Status = Ctrl(UsbIo, &DevReq, EfiUsbDataIn, 3000, Buf, &DataLen, &UsbStatus);

      if (!EFI_ERROR(Status) && DataLen > 0) {
        Info(L"[HubProbe] Handle %d: Hub descriptor type 0x%02x returned %d bytes (UsbStatus=%u):", (UINT32)h, dtype, (UINT32)DataLen, UsbStatus);
        PrintHexInlineSmall(Buf, DataLen);

        // If USB2 hub descriptor (0x29), parse a couple of fields safely
        if (dtype == 0x29 && DataLen >= 5) {
          UINT8 *b = (UINT8*)Buf;
          UINT8 bNbrPorts = b[2];
          // wHubChar is a little-endian UINT16 at offsets 3 (LSB) and 4 (MSB)
          UINT16 wHubChar = (UINT16)(b[3] | (b[4] << 8));
          Info(L"  bNbrPorts=%d wHubChar=0x%04x", (UINT32)bNbrPorts, (UINT32)wHubChar);
        }
      } else {
        Info(L"[HubProbe] Handle %d: Hub descriptor 0x%02x not present or failed (%r) UsbStatus=%u", (UINT32)h, dtype, Status, UsbStatus);
      }
      FreePool(Buf);
    } // types
  } // handles

  FreePool(Handles);
  Info(L"--- FetchHubDescriptors: end ---\n");
}



//
// Main entry
//
EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;
  UINTN HandleCount = 0;
  EFI_HANDLE *HandleBuffer = NULL;

  Print(L"\nUsbDiagApp: Starting safe USB diagnostic sequence (read-only by default)\n");

  // Enumerate all PCI IO handles so we can look for USB controllers
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status)) {
    Print(L"Failed to enumerate PCI handles: %r\n", Status);
    return Status;
  }

  Info(L"Total PCI handles found: %d", HandleCount);

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(HandleBuffer[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) {
      continue;
    }

    UINT8 classCode = 0, subClass = 0, progIf = 0;
    ReadPci8(PciIo, PCI_CFG_OFFSET_CLASSCODE, &classCode);
    ReadPci8(PciIo, PCI_CFG_OFFSET_SUBCLASS, &subClass);
    ReadPci8(PciIo, PCI_CFG_OFFSET_PROGIF, &progIf);

    // If this is a USB Host Controller class (0x0C/0x03), inspect further
    if (classCode == 0x0C && subClass == 0x03) {
      InspectUsbController(PciIo, classCode, subClass, progIf);
    }
  }

  if (HandleBuffer) {
    FreePool(HandleBuffer);
  }

  Info(L"UsbDiagApp: Completed checks. Press any key to exit (control won't block here). \n");


	// call before raw packet test
	DeepProbeUsbTopology();
	
	// monitor for a 5 seconds topology
	MonitorUsbTopologyForFiveSeconds();
	
	  Info(L"UsbDiagApp: starting async multi-listen (5s) — watch the console for raw packets");
	  UINTN started = StartAsyncMultiHandleListen(5);
	  if (started == 0) {
		Info(L"No async listeners started (no Interrupt-IN endpoints on published UsbIo handles).");
		TryExtraInvestigationsAfterAsyncListen();
	  } else {
		Info(L"Started %d async listeners (see callbacks printing raw packets).", started);
	  }

	// next phase 
	FetchHidReportDescriptors();
	FetchHubDescriptors();

	///////// this found only 1 handle and it's a USB flash /////////////
	// so this cannot read raw data packets like mouse device
#if 1  
  // read RAW packets from some particular devices like mouse
  Info(L" -- UsbDiagApp: prepare for read RAW packets --  \n");
  TestListenRawForFiveSecondsPerDevice();
  
  // at the end of UefiMain before exit:
  Info(L" -- UsbDiagApp: wait for packets for 5 seconds test \n --");
  //ListenRawFirstHandle(5); ///////////////////////////// OFF - this listen for USB flash so can't works without any interrupts 
#endif
  
  return EFI_SUCCESS;
}
