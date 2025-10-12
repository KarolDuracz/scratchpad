//
// Minimal UEFI app: detect PCI USB host controllers and try to initialize OHCI
// (Imitates enough of DXE OHCI initialization to set HcHCCA and bring HC to OP state).
//
// Build & run at your own risk. Tested conceptually; tweak for your platform/IOMMU settings.
//

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/DebugLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Protocol/PciIo.h>
#include <Library/BaseMemoryLib.h>


//
// OHCI register offsets and masks (from OHCI spec / kernel headers)
//
#define OHCI_REVISION           0x00
#define OHCI_CONTROL            0x04
#define OHCI_COMMAND_STATUS     0x08
#define OHCI_INTERRUPT_STATUS   0x0C
#define OHCI_INTERRUPT_ENABLE   0x10
#define OHCI_INTERRUPT_DISABLE  0x14
#define OHCI_HCCA               0x18
#define OHCI_CONTROL_HEAD_ED    0x20
#define OHCI_BULK_HEAD_ED       0x28
#define OHCI_DONE_HEAD          0x30
/* Root hub registers start around 0x48 */
#define OHCI_RHDESCRIPTORA      0x48
#define OHCI_RHPORTSTATUS_BASE  0x50  // port1 at 0x54? We'll compute below

/* HcControl bits */
#define OHCI_CTRL_CBSR  (3 << 0)
#define OHCI_CTRL_PLE   (1 << 2)
#define OHCI_CTRL_IE    (1 << 3)
#define OHCI_CTRL_CLE   (1 << 4)
#define OHCI_CTRL_BLE   (1 << 5)
#define OHCI_CTRL_HCFS  (3 << 6)    /* Host Controller Functional State */
#define OHCI_USB_RESET  (0 << 6)
#define OHCI_USB_RESUME (1 << 6)
#define OHCI_USB_OPER   (2 << 6)
#define OHCI_USB_SUSP   (3 << 6)

/* HcCommandStatus bits */
#define OHCI_HCR        (1 << 0) /* Host Controller Reset */

/* Root hub port status masks */
#define RH_PS_CCS       0x00000001 /* current connect status */
#define RH_PS_PES       0x00000002 /* port enable status */
#define RH_PS_PSS       0x00000004 /* port suspend status */
#define RH_PS_PRS       0x00000010 /* port reset status */
#define RH_PS_PPS       0x00000100 /* port power status */
#define RH_PS_LSDA      0x00000200 /* low speed device attached */

/* PCI class codes for USB */
#define PCI_CLASS_SERIAL_BUS 0x0C
#define PCI_SUBCLASS_USB     0x03
#define PCI_PROGIF_OHCI      0x10
#define PCI_PROGIF_EHCI      0x20
#define PCI_PROGIF_UHCI      0x00

//
// Helper: read/write OHCI MMIO via EFI_PCI_IO (barIndex 0 assumed for OHCI)
//
STATIC
EFI_STATUS
OhciRead32 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINT32              Offset,
  OUT UINT32             *Value
  )
{
  return PciIo->Mem.Read(
           PciIo,
           EfiPciIoWidthUint32,
           BarIndex,
           Offset,
           1,
           Value
         );
}

STATIC
EFI_STATUS
OhciWrite32 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINT32              Offset,
  IN UINT32              Value
  )
{
  return PciIo->Mem.Write(
           PciIo,
           EfiPciIoWidthUint32,
           BarIndex,
           Offset,
           1,
           &Value
         );
}

//
// Try to perform a minimal OHCI initialization.
// Returns EFI_SUCCESS if we managed to program an HCCA and put HC into OPERATIONAL (best-effort).
//
STATIC
EFI_STATUS
InitializeOhciController (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN EFI_HANDLE          PciHandle
  )
{
  EFI_STATUS Status;
  UINT32 HcRevision;
  UINT32 HcControl;
  UINT32 CmdStatus;
  UINT32 RhA;
  UINT32 NumPorts;
  EFI_PHYSICAL_ADDRESS HccaPhys = 0;
  VOID *HccaPtr = NULL;
  UINT8 BarIndex = 0; // OHCI uses BAR0
  UINTN PollLimit;

  Print(L"  Detected OHCI host controller. Trying minimal init...\n");

  // Read some registers
  Status = OhciRead32(PciIo, BarIndex, OHCI_REVISION, &HcRevision);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: failed to read REV (%r)\n", Status);
    return Status;
  }
  Print(L"    HcRevision = 0x%08x\n", HcRevision);

  Status = OhciRead32(PciIo, BarIndex, OHCI_CONTROL, &HcControl);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: failed to read CONTROL (%r)\n", Status);
    return Status;
  }
  Print(L"    HcControl = 0x%08x\n", HcControl);

  Status = OhciRead32(PciIo, BarIndex, OHCI_COMMAND_STATUS, &CmdStatus);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: failed to read CMDSTATUS (%r)\n", Status);
    return Status;
  }
  Print(L"    HcCommandStatus = 0x%08x\n", CmdStatus);

  // If HcHCCA is zero, we need to allocate HCCA and program it.
  {
    UINT32 HcHCCA;
    Status = OhciRead32(PciIo, BarIndex, OHCI_HCCA, &HcHCCA);
    if (EFI_ERROR(Status)) {
      Print(L"    OHCI: failed to read HCCA (%r)\n", Status);
      return Status;
    }
    Print(L"    HcHCCA = 0x%08x\n", HcHCCA);
  }

  //
  // Perform a soft reset (HCR) then program HcHCCA.
  // According to OHCI spec the reset must complete within ~10 ms.
  //
  Print(L"    Writing HcCommandStatus.HCR to reset controller...\n");
  Status = OhciWrite32(PciIo, BarIndex, OHCI_COMMAND_STATUS, OHCI_HCR);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: failed to write HCR (%r)\n", Status);
    return Status;
  }

  // Poll until HCR clears (or timeout)
  PollLimit = 1000; // poll attempt count (with small stalls) -> ~1000 * 10us = 10ms
  while (PollLimit--) {
    Status = OhciRead32(PciIo, BarIndex, OHCI_COMMAND_STATUS, &CmdStatus);
    if (EFI_ERROR(Status)) {
      Print(L"    OHCI: read CMDSTATUS failed while waiting for HCR to clear (%r)\n", Status);
      return Status;
    }
    if ((CmdStatus & OHCI_HCR) == 0) {
      break;
    }
    gBS->Stall(10); // 10 microseconds
  }
  if (PollLimit == 0) {
    Print(L"    OHCI: reset did not complete in time (continuing best-effort)\n");
    // Not necessarily fatal; continue best-effort
  } else {
    Print(L"    OHCI: reset complete\n");
  }

  //
  // Allocate a page for HCCA (256-byte structure) - use AllocatePages so we have a physical address
  //
  Status = gBS->AllocatePages(AllocateAnyPages, EfiBootServicesData, 1, &HccaPhys);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: AllocatePages for HCCA failed (%r)\n", Status);
    return Status;
  }
  HccaPtr = (VOID *)(UINTN)HccaPhys;
  ZeroMem(HccaPtr, EFI_PAGE_SIZE);
  Print(L"    Allocated HCCA at phys 0x%p\n", (VOID*)(UINTN)HccaPhys);

  // Program HcHCCA (low 32 bits). Note: on 64-bit platforms an OHCI controller may only accept 32-bit addresses.
  Status = OhciWrite32(PciIo, BarIndex, OHCI_HCCA, (UINT32)(UINTN)HccaPhys);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: failed to write HcHCCA (%r)\n", Status);
    // free allocated pages
    gBS->FreePages(HccaPhys, 1);
    return Status;
  }

  // Clear Control/Bulk head ED pointers and done head to zero for safety
  (void)OhciWrite32(PciIo, BarIndex, OHCI_CONTROL_HEAD_ED, 0);
  (void)OhciWrite32(PciIo, BarIndex, OHCI_BULK_HEAD_ED, 0);
  (void)OhciWrite32(PciIo, BarIndex, OHCI_DONE_HEAD, 0);

  // Set HC functional state to OPERATIONAL (OHCI_USB_OPER) — keep other control bits as-is except HCFS
  HcControl &= ~OHCI_CTRL_HCFS;            // clear HCFS field
  HcControl |= OHCI_USB_OPER;             // set OPERATIONAL
  Status = OhciWrite32(PciIo, BarIndex, OHCI_CONTROL, HcControl);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: failed to set HcControl to OP (%r)\n", Status);
    gBS->FreePages(HccaPhys, 1);
    return Status;
  }
  Print(L"    HcControl set to OPERATIONAL (0x%08x)\n", HcControl);

  //
  // Read root hub descriptor A to find number of ports and report per-port status
  //
  Status = OhciRead32(PciIo, BarIndex, OHCI_RHDESCRIPTORA, &RhA);
  if (EFI_ERROR(Status)) {
    Print(L"    OHCI: failed to read RhDescriptorA (%r)\n", Status);
    gBS->FreePages(HccaPhys, 1);
    return Status;
  }
  NumPorts = RhA & 0xFF;
  Print(L"    Root hub: %u downstream ports (RhDescriptorA=0x%08x)\n", NumPorts, RhA);

  if (NumPorts > 0 && NumPorts < 32) {
    for (UINT32 i = 1; i <= NumPorts; ++i) {
      UINT32 PortStatus;
      UINT32 PortOffset = OHCI_RHPORTSTATUS_BASE + (i - 1) * 4; // port1 at base, port2 at base+4, ...
      Status = OhciRead32(PciIo, BarIndex, PortOffset, &PortStatus);
      if (EFI_ERROR(Status)) {
        Print(L"      Port %u: read failed (%r)\n", i, Status);
        continue;
      }
      Print(L"      Port %u: Status=0x%08x  Connected=%u  Enabled=%u  Powered=%u  LowSpeed=%u\n",
            i,
            PortStatus,
            !!(PortStatus & RH_PS_CCS),
            !!(PortStatus & RH_PS_PES),
            !!(PortStatus & RH_PS_PPS),
            !!(PortStatus & RH_PS_LSDA)
      );
    }
  } else {
    Print(L"    Root hub reports unreasonable port count %u, skipping port read\n", NumPorts);
  }

  // Keep HCCA allocated as long as app runs — free at exit in this demo
  // Alternatively, free now and leave controller unconfigured:
  // gBS->FreePages(HccaPhys, 1);

  return EFI_SUCCESS;
}

//
// Main entry: scan PCI, locate USB host controllers and try to init OHCI devices.
//
EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN HandleCount = 0;

  Print(L"\nScanning PCI devices for USB host controllers...\n\n");

  // Locate all PCI_IO handles
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &PciHandles);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Print(L"No PCI devices found or LocateHandleBuffer failed: %r\n", Status);
    return EFI_SUCCESS;
  }

  for (UINTN idx = 0; idx < HandleCount; ++idx) {
    EFI_PCI_IO_PROTOCOL *PciIo;
    Status = gBS->HandleProtocol(PciHandles[idx], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status)) {
      continue;
    }

    // Read PCI vendor/device
    UINT32 DevV;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x00, 1, &DevV);
    if (EFI_ERROR(Status)) {
      continue;
    }
    UINT16 VendorId = (UINT16)(DevV & 0xFFFF);
    UINT16 DeviceId = (UINT16)((DevV >> 16) & 0xFFFF);

    // Read class/subclass/progIF at offset 0x08
    UINT32 ClassReg;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassReg);
    if (EFI_ERROR(Status)) {
      continue;
    }
    UINT8 BaseClass = (UINT8)((ClassReg >> 24) & 0xFF);
    UINT8 SubClass  = (UINT8)((ClassReg >> 16) & 0xFF);
    UINT8 ProgIf    = (UINT8)((ClassReg >> 8) & 0xFF);

    if (BaseClass == PCI_CLASS_SERIAL_BUS && SubClass == PCI_SUBCLASS_USB) {
      // This is a USB host controller
      // Read BAR0 for info
      UINT32 Bar0;
      PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x10, 1, &Bar0);

      // Read PCI location (use PciIo->Pci.Read of bus/device? Usually via attributes)
      // We can print handle pointer and vendor/device/progIF
      Print(L"PCI USB Host Controller found (handle=0x%p): Class=0x%02x Sub=0x%02x ProgIf=0x%02x\n",
            PciHandles[idx], BaseClass, SubClass, ProgIf);
      Print(L"  PCI VendorId=0x%04x DeviceId=0x%04x\n", VendorId, DeviceId);
      Print(L"  BAR0 (raw) = 0x%08x\n", Bar0);

      // If OHCI, attempt minimal initialization
      if (ProgIf == PCI_PROGIF_OHCI) {
        InitializeOhciController(PciIo, PciHandles[idx]);
      } else {
        Print(L"  (Non-OHCI host controller; skipping software init)\n");
      }

      Print(L"\n");
    }
  }

  // wait for key press
  Print(L"Done. Press any key to exit...\n");
  EFI_INPUT_KEY Key;
  while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) != EFI_SUCCESS) {
    // spin
  }

  if (PciHandles) {
    FreePool(PciHandles);
  }

  return EFI_SUCCESS;
}
