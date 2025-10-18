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

// GOP
#include <Protocol/GraphicsOutput.h>

static EFI_GRAPHICS_OUTPUT_PROTOCOL *mGraphicsOuput = NULL;

EFI_GRAPHICS_OUTPUT_BLT_PIXEL white = { 255, 255, 255, 0 };
EFI_GRAPHICS_OUTPUT_BLT_PIXEL blue = { 255, 234, 0, 0 };

// others
#include <Protocol/DevicePath.h>
//#include <Protocol/PciIo.h>
//#include <Protocol/Usb2HostController.h>


// USB enumerate
// ===== add/ensure these includes near the top =====
#include <Protocol/Usb2HostController.h>
#include <Protocol/UsbIo.h>
#include <IndustryStandard/Usb.h>

// ===== local fallbacks / constants =====
#ifndef USB_PORT_FEAT_RESET
#define USB_PORT_FEAT_RESET     4
#endif

#ifndef USB_PORT_FEAT_C_RESET
#define USB_PORT_FEAT_C_RESET   20
#endif

#ifndef USB_PORT_STAT_CONNECTION
#define USB_PORT_STAT_CONNECTION   0x0001
#endif
#ifndef USB_PORT_STAT_ENABLE
#define USB_PORT_STAT_ENABLE       0x0002
#endif
#ifndef USB_PORT_STAT_RESET
#define USB_PORT_STAT_RESET        0x0010
#endif
#ifndef USB_PORT_STAT_LOW_SPEED
#define USB_PORT_STAT_LOW_SPEED    0x0200
#endif
#ifndef USB_PORT_STAT_C_RESET
#define USB_PORT_STAT_C_RESET      (1u << 20)  // 0x0010 0000
#endif

#define USB_REQ_GET_DESCRIPTOR    0x06
#define USB_REQ_SET_ADDRESS       0x05
#define USB_REQ_SET_CONFIGURATION 0x09
#define USB_REQ_GET_STATUS        0x00
#define USB_REQ_SET_FEATURE       0x03
#define USB_REQ_CLEAR_FEATURE     0x01

#define USB_DESC_TYPE_DEVICE      0x01
#define USB_DESC_TYPE_CONFIG      0x02
#define USB_DESC_TYPE_HUB         0x29

#define HUB_CLASS                 0x09

// ===== forward declarations =====
STATIC EFI_STATUS EnumerateRootHub(IN EFI_USB2_HC_PROTOCOL *Usb2Hc, IN UINT32 NumPorts, IN OUT UINT8 *NextUsbAddr);
STATIC EFI_STATUS EnumerateDeviceRecursive(IN EFI_USB2_HC_PROTOCOL *Usb2Hc, IN UINT8 DeviceSpeed, IN UINT8 DeviceAddress, IN UINTN MaxPacketLen, IN UINTN Depth, IN OUT UINT8 *NextUsbAddr);
STATIC EFI_STATUS EnumerateHubPorts(IN EFI_USB2_HC_PROTOCOL *Usb2Hc, IN UINT8 HubAddr, IN UINT8 HubSpeed, IN UINTN HubMaxPacket, IN UINTN Depth, IN OUT UINT8 *NextUsbAddr);



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


/*
  Helper: call the host controller ControlTransfer with the exact parameters
  that the EDK2 header expects. Translator is optional (we pass NULL for
  simple cases).
*/
// ===== wrapper to call ControlTransfer with correct param order/types =====
/*
STATIC
EFI_STATUS
CallControlTransfer (
  IN EFI_USB2_HC_PROTOCOL       *Usb2Hc,
  IN UINT8                      DeviceAddress,
  IN UINT8                      DeviceSpeed,
  IN UINTN                      MaxPacket,
  IN EFI_USB_DEVICE_REQUEST     *Req,
  IN EFI_USB_DATA_DIRECTION     Dir,
  IN OUT VOID                  *Data,
  IN OUT UINTN                 *DataLength,
  IN UINTN                      TimeoutMs,
  OUT UINT32                   *TransferResult OPTIONAL
  )
{
  if (Usb2Hc == NULL || Req == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  //
  // EDK2 prototype:
  //   ControlTransfer(
  //     This,
  //     DeviceAddress,
  //     DeviceSpeed,
  //     MaxPacket,
  //     Request,
  //     TransferDirection,
  //     Data,
  //     DataLength,
  //     Translator,   // EFI_USB2_HC_TRANSACTION_TRANSLATOR *
  //     Timeout,
  //     TransferResult
  //   );
  //
  // We pass NULL for Translator for simple control transfers.
  //
  return Usb2Hc->ControlTransfer(
           Usb2Hc,
           DeviceAddress,
           DeviceSpeed,
           MaxPacket,
           Req,
           Dir,
           Data,
           DataLength,
           NULL,             // EFI_USB2_HC_TRANSACTION_TRANSLATOR *
           TimeoutMs,
           TransferResult
         );
}
*/
// ---------- CallControlTransfer wrapper (matches the exact EDK2 prototype) ----------
STATIC
EFI_STATUS
CallControlTransfer (
  IN EFI_USB2_HC_PROTOCOL       *Usb2Hc,
  IN UINT8                      DeviceAddress,
  IN UINT8                      DeviceSpeed,
  IN UINTN                      MaximumPacketLength,
  IN EFI_USB_DEVICE_REQUEST     *Request,
  IN EFI_USB_DATA_DIRECTION     TransferDirection,
  IN OUT VOID                  *Data OPTIONAL,
  IN OUT UINTN                 *DataLength OPTIONAL,
  IN UINTN                      TimeOut,
  OUT UINT32                   *TransferResult OPTIONAL
  )
{
  if (Usb2Hc == NULL || Request == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  //
  // IMPORTANT: The EDK2 prototype is:
  //   ControlTransfer(
  //     This,
  //     DeviceAddress,
  //     DeviceSpeed,
  //     MaximumPacketLength,
  //     Request,
  //     TransferDirection,
  //     Data,
  //     DataLength,
  //     TimeOut,                 <-- TIMEOUT before TRANSLATOR
  //     Translator,              <-- TRANSLATOR pointer here
  //     TransferResult
  //   );
  //
  // For simple cases we pass NULL for Translator.
  //
  return Usb2Hc->ControlTransfer(
           Usb2Hc,
           DeviceAddress,
           DeviceSpeed,
           MaximumPacketLength,
           Request,
           TransferDirection,
           Data,
           DataLength,
           TimeOut,               // TIMEOUT (UINTN)
           NULL,                  // EFI_USB2_HC_TRANSACTION_TRANSLATOR *
           TransferResult
         );
}

/*

	// small routine for this part

	EFI_HANDLE *FoundUsbHandle = NULL;
	EFI_USB2_HC_PROTOCOL *Usb2Hc = NULL;
	EFI_STATUS s = FindUsb2HcForPciHandle(PciHandle, PciIo, &FoundUsbHandle, &Usb2Hc);
	if (!EFI_ERROR(s) && Usb2Hc != NULL) {
	  Print(L"    Using EFI_USB2_HC_PROTOCOL from handle %p (protocol at %p)\n", FoundUsbHandle ? *FoundUsbHandle : NULL, Usb2Hc);
	  UINT8 nextAddr = 1;
	  (VOID)EnumerateRootHub(Usb2Hc, NumPorts, &nextAddr);
	  // free FoundUsbHandle if you allocated it
	  if (FoundUsbHandle) { FreePool(FoundUsbHandle); }
	} else {
	  Print(L"    Could not find matching EFI_USB2_HC_PROTOCOL for this PCI handle (will skip).\n");
	}


*/
STATIC
EFI_STATUS
FindUsb2HcForPciHandle(
  IN  EFI_HANDLE            PciHandle,
  IN  EFI_PCI_IO_PROTOCOL  *PciIoOriginal, // you already have this for the handle
  OUT EFI_HANDLE          **MatchingUsbHcHandleOut, // allocate NULL or pointer on stack
  OUT EFI_USB2_HC_PROTOCOL **Usb2HcOut // returned protocol pointer (not opened)
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;
  EFI_HANDLE FoundHandle = NULL;
  EFI_USB2_HC_PROTOCOL *FoundUsb2Hc = NULL;

  *MatchingUsbHcHandleOut = NULL;
  *Usb2HcOut = NULL;

  // read original PCI device id/class for comparison
  UINT32 OrigDevV = 0, OrigClassReg = 0;
  Status = PciIoOriginal->Pci.Read(PciIoOriginal,
                                   EfiPciIoWidthUint32,
                                   0x00,
                                   1,
                                   &OrigDevV);
  if (EFI_ERROR(Status)) {
    return Status;
  }
  Status = PciIoOriginal->Pci.Read(PciIoOriginal,
                                   EfiPciIoWidthUint32,
                                   0x08,
                                   1,
                                   &OrigClassReg);
  if (EFI_ERROR(Status)) {
    return Status;
  }

  // Find all handles that implement EFI_USB2_HC_PROTOCOL
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &Count, &Handles);
  if (EFI_ERROR(Status) || Count == 0) {
    // none found
    return EFI_NOT_FOUND;
  }

  for (UINTN i = 0; i < Count; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIoCandidate = NULL;
    // Try to open PCI IO on the candidate handle (not all child handles have it)
    Status = gBS->HandleProtocol(Handles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIoCandidate);
    if (!EFI_ERROR(Status) && PciIoCandidate != NULL) {
      // read vendor/device/class from candidate and compare
      UINT32 CandDevV = 0, CandClass = 0;
      if (!EFI_ERROR(PciIoCandidate->Pci.Read(PciIoCandidate, EfiPciIoWidthUint32, 0x00, 1, &CandDevV)) &&
          !EFI_ERROR(PciIoCandidate->Pci.Read(PciIoCandidate, EfiPciIoWidthUint32, 0x08, 1, &CandClass))) {
        if (CandDevV == OrigDevV && CandClass == OrigClassReg) {
          // match!
          FoundHandle = Handles[i];
          // open the USB2_HC protocol on it
          Status = gBS->HandleProtocol(FoundHandle, &gEfiUsb2HcProtocolGuid, (VOID**)&FoundUsb2Hc);
          if (EFI_ERROR(Status) || FoundUsb2Hc == NULL) {
            // weird but continue scanning
            FoundHandle = NULL;
            FoundUsb2Hc = NULL;
            continue;
          }
          break;
        }
      }
    }
  }

  // fallback: if not found by PCI match, pick first usable usb2hc handle
  if (FoundHandle == NULL) {
    for (UINTN i = 0; i < Count; ++i) {
      EFI_USB2_HC_PROTOCOL *ProbeUsb = NULL;
      if (!EFI_ERROR(gBS->HandleProtocol(Handles[i], &gEfiUsb2HcProtocolGuid, (VOID**)&ProbeUsb)) && ProbeUsb != NULL) {
        FoundHandle = Handles[i];
        FoundUsb2Hc = ProbeUsb;
        break;
      }
    }
  }

  if (Handles) {
    FreePool(Handles);
  }

  if (FoundHandle != NULL) {
    *MatchingUsbHcHandleOut = AllocatePool(sizeof(EFI_HANDLE));
    if (*MatchingUsbHcHandleOut == NULL) {
      return EFI_OUT_OF_RESOURCES;
    }
    **MatchingUsbHcHandleOut = FoundHandle;
    *Usb2HcOut = FoundUsb2Hc;
    return EFI_SUCCESS;
  }

  return EFI_NOT_FOUND;
}



//
// Helper: print an EFI_GUID in human readable form
//
STATIC
VOID
PrintGuid (
  IN CONST EFI_GUID *Guid
  )
{
  if (Guid == NULL) {
    Print(L"(null guid)\n");
    return;
  }
  // Print as 01234567-89ab-cdef-0123-456789abcdef
  Print(L"%08x-%04x-%04x-", Guid->Data1, Guid->Data2, Guid->Data3);
  for (UINTN i = 0; i < 2; ++i) {
    Print(L"%02x", Guid->Data4[i]);
  }
  Print(L"-");
  for (UINTN i = 2; i < 8; ++i) {
    Print(L"%02x", Guid->Data4[i]);
  }
  Print(L"\n");
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
  
  
  
  
  // NOT HERE -> plase it in EnumerateRootHub
	  	  // --- after NumPorts computed and printed ---
		  
		  EFI_USB2_HC_PROTOCOL *Usb2Hc = NULL;
		  Status = gBS->HandleProtocol(PciHandle, &gEfiUsb2HcProtocolGuid, (VOID**)&Usb2Hc);
		  if (!EFI_ERROR(Status) && Usb2Hc != NULL) {
			Print(L"    Found EFI_USB2_HC_PROTOCOL; launching enumerator...\n");
			UINT8 nextAddr = 1;
			(VOID)EnumerateRootHub(Usb2Hc, NumPorts, &nextAddr);
		  } else {
			Print(L"    EFI_USB2_HC_PROTOCOL not found on this handle - high-level enumeration skipped.\n");
			
			
			  //
		// Diagnostic block to paste where you currently print the "not found" message.
		// Place it immediately after this check/print in InitializeOhciController():
		//
		//   Status = gBS->HandleProtocol(PciHandle, &gEfiUsb2HcProtocolGuid, (VOID**)&Usb2Hc);
		//   if (!EFI_ERROR(Status) && Usb2Hc != NULL) { ... }
		//   else {
		//     Print(L"    EFI_USB2_HC_PROTOCOL not found on this handle - high-level enumeration skipped.\n");
		//     // <-- paste diagnostic block here
		//   }
		//
		{
		  EFI_STATUS DiagStatus;

		  Print(L"    Starting diagnostics for handle 0x%p\n", PciHandle);

		  // 1) List protocols installed ON THIS HANDLE
		  EFI_GUID **ProtocolList = NULL;
		  UINTN ProtocolCount = 0;
		  DiagStatus = gBS->ProtocolsPerHandle(PciHandle, &ProtocolList, &ProtocolCount);
		  if (EFI_ERROR(DiagStatus)) {
			Print(L"    ProtocolsPerHandle() failed on this handle: %r\n", DiagStatus);
		  } else {
			Print(L"    Protocols installed on this handle: %u\n", (unsigned)ProtocolCount);
			for (UINTN ij = 0; ij < ProtocolCount; ++ij) {
			  Print(L"      [%02u] ", (unsigned)ij);
			  PrintGuid(ProtocolList[ij]);
			}
			// Caller must free ProtocolList per spec
			FreePool(ProtocolList);
		  }

		  // 2) Locate all handles that DO have EFI_USB2_HC_PROTOCOL and print them
		  EFI_HANDLE *UsbHcHandles = NULL;
		  UINTN UsbHcCount = 0;
		  DiagStatus = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &UsbHcCount, &UsbHcHandles);
		  if (EFI_ERROR(DiagStatus)) {
			Print(L"    LocateHandleBuffer(ByProtocol, EFI_USB2_HC_PROTOCOL) failed: %r\n", DiagStatus);
		  } else {
			Print(L"    System reports %u handle(s) with EFI_USB2_HC_PROTOCOL\n", (unsigned)UsbHcCount);
			for (UINTN ik = 0; ik < UsbHcCount; ++ik) {
			  EFI_HANDLE h = UsbHcHandles[ik];
			  Print(L"      Handle[%u] = 0x%p", (unsigned)ik, h);
			  if (h == PciHandle) {
				Print(L"  <-- same as this PCI handle\n");
			  } else {
				Print(L"\n");
			  }

			  // Try to open the protocol on that handle to confirm it's accessible
			  EFI_USB2_HC_PROTOCOL *ProbeUsb2Hc = NULL;
			  DiagStatus = gBS->HandleProtocol(h, &gEfiUsb2HcProtocolGuid, (VOID**)&ProbeUsb2Hc);
			  if (EFI_ERROR(DiagStatus) || ProbeUsb2Hc == NULL) {
				Print(L"        Couldn't open EFI_USB2_HC_PROTOCOL on that handle: %r\n", DiagStatus);
			  } else {
				Print(L"        EFI_USB2_HC_PROTOCOL open OK, pointer=0x%p\n", ProbeUsb2Hc);
				// If you want, print a couple of pointer-sized fields from the protocol to help debugging:
				// (don't dereference function pointers, just show pointer values)
				Print(L"        Protocol struct address: 0x%p\n", ProbeUsb2Hc);
				// Optionally inspect function pointers (addresses)
				Print(L"        ControlTransfer@%p  GetRootHubPortStatus@%p\n",
					  ProbeUsb2Hc->ControlTransfer, ProbeUsb2Hc->GetRootHubPortStatus);
			  }
			}
			FreePool(UsbHcHandles);
		  }

		  // 3) Try to locate the USB2_HC protocol anywhere in the system by GUID name (LocateProtocol)
		  EFI_USB2_HC_PROTOCOL *AnyUsb2Hc = NULL;
		  DiagStatus = gBS->LocateProtocol(&gEfiUsb2HcProtocolGuid, NULL, (VOID**)&AnyUsb2Hc);
		  if (!EFI_ERROR(DiagStatus) && AnyUsb2Hc != NULL) {
			Print(L"    LocateProtocol found an EFI_USB2_HC_PROTOCOL instance at %p (but maybe on different handle)\n", AnyUsb2Hc);
		  } else {
			Print(L"    LocateProtocol did not find EFI_USB2_HC_PROTOCOL (Status=%r). Host controller driver may not be loaded.\n", DiagStatus);
		  }

		  Print(L"    End diagnostics for handle 0x%p\n", PciHandle);
		}
		// diagnostic end
			
		  }
  
  
  // after 
  EFI_HANDLE *FoundUsbHandle = NULL;
EFI_USB2_HC_PROTOCOL *Usb2Hc_test = NULL;
EFI_STATUS s = FindUsb2HcForPciHandle(PciHandle, PciIo, &FoundUsbHandle, &Usb2Hc_test);
if (!EFI_ERROR(s) && Usb2Hc_test != NULL) {
  Print(L"    Using EFI_USB2_HC_PROTOCOL from handle %p (protocol at %p)\n", FoundUsbHandle ? *FoundUsbHandle : NULL, Usb2Hc_test);
  UINT8 nextAddr = 1;
  (VOID)EnumerateRootHub(Usb2Hc_test, NumPorts, &nextAddr);
  // free FoundUsbHandle if you allocated it
  if (FoundUsbHandle) { FreePool(FoundUsbHandle); }
} else {
  Print(L"    Could not find matching EFI_USB2_HC_PROTOCOL for this PCI handle (will skip).\n");
}
	// end
  

  return EFI_SUCCESS;
}



// stuff for enumerate
// ===== Enumerate root hub ports (uses EFI_USB2_HC_PROTOCOL) =====
STATIC
EFI_STATUS
EnumerateRootHub (
  IN EFI_USB2_HC_PROTOCOL *Usb2Hc,
  IN UINT32               NumPorts,
  IN OUT UINT8           *NextUsbAddr
  )
{
  EFI_STATUS Status;
  UINT8 devSpeed;
  for (UINT32 port = 0; port < NumPorts; ++port) {
    EFI_USB_PORT_STATUS PortStatus;
    Status = Usb2Hc->GetRootHubPortStatus(Usb2Hc, (UINT8)port, &PortStatus);
    if (EFI_ERROR(Status)) {
      Print(L"  GetRootHubPortStatus(port %u) failed: %r\n", (unsigned)port, Status);
      continue;
    }

    Print(L"  Root port %u: PortStatus=0x%08x\n", (unsigned)(port + 1), PortStatus.PortStatus);


    if (PortStatus.PortStatus & USB_PORT_STAT_CONNECTION) {
      // Reset root hub port
      Status = Usb2Hc->SetRootHubPortFeature(Usb2Hc, (UINT8)port, EfiUsbPortReset);
      if (EFI_ERROR(Status)) {
        Print(L"    Root port %u: SetRootHubPortFeature(RESET) failed: %r\n", (unsigned)(port + 1), Status);
        continue;
      }

      // Poll for reset completion: wait until reset bit is cleared or enabled
      UINTN retry = 100; // ~1s with 10ms sleep
      BOOLEAN reset_ok = FALSE;
      while (retry--) {
        gBS->Stall(10000); // 10ms
        Status = Usb2Hc->GetRootHubPortStatus(Usb2Hc, (UINT8)port, &PortStatus);
        if (EFI_ERROR(Status)) break;
        if (!(PortStatus.PortStatus & USB_PORT_STAT_RESET)) {
          reset_ok = TRUE;
          break;
        }
      }
      if (!reset_ok) {
        Print(L"    Root port %u: reset timed out\n", (unsigned)(port + 1));
        continue;
      }

      // Determine device speed
      if (PortStatus.PortStatus & USB_PORT_STAT_LOW_SPEED) {
        devSpeed = EFI_USB_SPEED_LOW;
      } else {
        devSpeed = EFI_USB_SPEED_FULL;
      }

      // Assign address (take the value, then increment)
      UINT8 assigned = (UINT8)(*NextUsbAddr);
      (*NextUsbAddr) = (UINT8)((*NextUsbAddr) + 1);

      EFI_USB_DEVICE_REQUEST SetAddrReq;
      ZeroMem(&SetAddrReq, sizeof(SetAddrReq));
      SetAddrReq.RequestType = 0x00; // Host-to-device, Standard, Device
      SetAddrReq.Request = USB_REQ_SET_ADDRESS;
      SetAddrReq.Value = assigned;
      SetAddrReq.Index = 0;
      SetAddrReq.Length = 0;
      UINTN dummyLen = 0;
      UINT32 tr = 0;
      Status = CallControlTransfer(Usb2Hc, 0 /* addr 0 for SET_ADDRESS */, devSpeed, 8, &SetAddrReq, EfiUsbNoData, NULL, &dummyLen, 500, &tr);
      if (EFI_ERROR(Status)) {
        Print(L"    Root port %u: SET_ADDRESS(%u) failed: %r\n", (unsigned)(port + 1), (unsigned)assigned, Status);
        continue;
      }

      // Wait 2 ms per USB spec before using new address
      gBS->Stall(2000);

      // Recursively enumerate the newly addressed device
      Status = EnumerateDeviceRecursive(Usb2Hc, devSpeed, assigned, 8, 1, NextUsbAddr);
      if (EFI_ERROR(Status)) {
        Print(L"    Root port %u: child enumeration failed: %r\n", (unsigned)(port + 1), Status);
      }
    }
  }
  return EFI_SUCCESS;
}

// ===== Enumerate a hub's ports using class requests =====
STATIC
EFI_STATUS
EnumerateHubPorts (
  IN EFI_USB2_HC_PROTOCOL *Usb2Hc,
  IN UINT8                HubAddr,
  IN UINT8                HubSpeed,
  IN UINTN                HubMaxPacket,
  IN UINTN                Depth,
  IN OUT UINT8           *NextUsbAddr
  )
{
  EFI_STATUS Status;
  UINT8 HubDescBuf[256];
  UINTN Len = sizeof(HubDescBuf);
  EFI_USB_DEVICE_REQUEST Req;
  UINT32 TransferResult;

  // Try class GET_DESCRIPTOR for hub (bmRequestType 0xA0 then 0xA3 fallback)
  ZeroMem(&Req, sizeof(Req));
  Req.RequestType = 0xA0;
  Req.Request = USB_REQ_GET_DESCRIPTOR;
  Req.Value = (UINT16)((USB_DESC_TYPE_HUB << 8) | 0);
  Req.Index = 0;
  Req.Length = (UINT16)Len;

  Len = Req.Length;
  Status = CallControlTransfer(Usb2Hc, HubAddr, HubSpeed, HubMaxPacket, &Req, EfiUsbDataIn, HubDescBuf, &Len, 500, &TransferResult);
  if (EFI_ERROR(Status)) {
    // fallback bmRequestType
    Req.RequestType = 0xA3;
    Len = sizeof(HubDescBuf);
    Status = CallControlTransfer(Usb2Hc, HubAddr, HubSpeed, HubMaxPacket, &Req, EfiUsbDataIn, HubDescBuf, &Len, 500, &TransferResult);
    if (EFI_ERROR(Status)) {
      Print(L"%*sHub(addr %u): GetHubDescriptor failed: %r (tr=0x%x)\n", (int)(Depth * 2), L"", (unsigned)HubAddr, Status, TransferResult);
      return Status;
    }
  }

  if (Len < 4) {
    Print(L"%*sHub(addr %u): hub descriptor too short (%u)\n", (int)(Depth * 2), L"", (unsigned)HubAddr, (unsigned)Len);
    return EFI_DEVICE_ERROR;
  }

  UINT8 bNbrPorts = HubDescBuf[2];
  Print(L"%*sHub(addr %u): %u ports\n", (int)(Depth * 2), L"", (unsigned)HubAddr, (unsigned)bNbrPorts);

  for (UINT8 p = 1; p <= bNbrPorts; ++p) {
    // GET_PORT_STATUS (class, device-to-host, recipient=other => bmRequestType 0xA3)
    ZeroMem(&Req, sizeof(Req));
    Req.RequestType = 0xA3;
    Req.Request = USB_REQ_GET_STATUS;
    Req.Value = 0;
    Req.Index = p;
    Req.Length = 4;
    UINT8 PortStBuf[4];
    UINTN Plen = 4;
    Status = CallControlTransfer(Usb2Hc, HubAddr, HubSpeed, HubMaxPacket, &Req, EfiUsbDataIn, PortStBuf, &Plen, 500, &TransferResult);
    if (EFI_ERROR(Status) || Plen < 4) {
      Print(L"%*s  Port %u: GET_PORT_STATUS failed (%r)\n", (int)(Depth * 2), L"", (unsigned)p, Status);
      continue;
    }

    UINT32 portStatus = (UINT32)PortStBuf[0] | ((UINT32)PortStBuf[1] << 8);
    UINT32 portChange = (UINT32)PortStBuf[2] | ((UINT32)PortStBuf[3] << 8);
    Print(L"%*s  Port %u: status=0x%04x change=0x%04x\n", (int)(Depth * 2), L"", (unsigned)p, (unsigned)portStatus, (unsigned)portChange);

    if (portStatus & USB_PORT_STAT_CONNECTION) {
      // SET_FEATURE PORT_RESET (bmRequestType 0x23)
      ZeroMem(&Req, sizeof(Req));
      Req.RequestType = 0x23;
      Req.Request = USB_REQ_SET_FEATURE;
      Req.Value = (UINT16)USB_PORT_FEAT_RESET;
      Req.Index = p;
      Req.Length = 0;
      UINTN zeroLen = 0;
      Status = CallControlTransfer(Usb2Hc, HubAddr, HubSpeed, HubMaxPacket, &Req, EfiUsbNoData, NULL, &zeroLen, 500, &TransferResult);
      if (EFI_ERROR(Status)) {
        Print(L"%*s    Port %u: SetPortFeature(RESET) failed (%r)\n", (int)(Depth * 2), L"", (unsigned)p, Status);
        continue;
      }

      // Poll for reset completion
      UINTN retry = 100;
      BOOLEAN reset_ok = FALSE;
      while (retry--) {
        gBS->Stall(10000); // 10 ms
        Plen = 4;
        ZeroMem(&Req, sizeof(Req));
        Req.RequestType = 0xA3;
        Req.Request = USB_REQ_GET_STATUS;
        Req.Value = 0;
        Req.Index = p;
        Req.Length = 4;
        Status = CallControlTransfer(Usb2Hc, HubAddr, HubSpeed, HubMaxPacket, &Req, EfiUsbDataIn, PortStBuf, &Plen, 500, &TransferResult);
        if (EFI_ERROR(Status)) break;
        portStatus = (UINT32)PortStBuf[0] | ((UINT32)PortStBuf[1] << 8);
        if (!(portStatus & USB_PORT_STAT_RESET)) {
          reset_ok = TRUE;
          // Clear change bit for reset (class CLEAR_FEATURE C_PORT_RESET)
          ZeroMem(&Req, sizeof(Req));
          Req.RequestType = 0x23;
          Req.Request = USB_REQ_CLEAR_FEATURE;
          Req.Value = (UINT16)USB_PORT_FEAT_C_RESET;
          Req.Index = p;
          Req.Length = 0;
          zeroLen = 0;
          (void)CallControlTransfer(Usb2Hc, HubAddr, HubSpeed, HubMaxPacket, &Req, EfiUsbNoData, NULL, &zeroLen, 200, &TransferResult);
          break;
        }
      }

      if (!reset_ok) {
        Print(L"%*s    Port %u: reset timed out\n", (int)(Depth * 2), L"", (unsigned)p);
        continue;
      }

      // Assign address to device on hub port
      UINT8 assigned = (UINT8)(*NextUsbAddr);
      (*NextUsbAddr) = (UINT8)((*NextUsbAddr) + 1);

      ZeroMem(&Req, sizeof(Req));
      Req.RequestType = 0x00;
      Req.Request = USB_REQ_SET_ADDRESS;
      Req.Value = assigned;
      Req.Index = 0;
      Req.Length = 0;
      UINTN z = 0;
      Status = CallControlTransfer(Usb2Hc, 0 /* addr 0 */, HubSpeed, 8, &Req, EfiUsbNoData, NULL, &z, 500, &TransferResult);
      if (EFI_ERROR(Status)) {
        Print(L"%*s    Port %u: SET_ADDRESS(%u) failed (%r)\n", (int)(Depth * 2), L"", (unsigned)p, (unsigned)assigned, Status);
        continue;
      }
      gBS->Stall(2000); // 2 ms

      Status = EnumerateDeviceRecursive(Usb2Hc, HubSpeed, assigned, 8, Depth + 1, NextUsbAddr);
      if (EFI_ERROR(Status)) {
        Print(L"%*s    Port %u: child enum failed (%r)\n", (int)(Depth * 2), L"", (unsigned)p, Status);
      }
    }
  }

  return EFI_SUCCESS;
}

// ===== Enumerate a device (GET_DEVICE_DESCRIPTOR, GET_CONFIG, SET_CONFIGURATION, hub detection) =====
STATIC
EFI_STATUS
EnumerateDeviceRecursive (
  IN EFI_USB2_HC_PROTOCOL *Usb2Hc,
  IN UINT8                DeviceSpeed,
  IN UINT8                DeviceAddress,
  IN UINTN                MaxPacketLen,
  IN UINTN                Depth,
  IN OUT UINT8           *NextUsbAddr
  )
{
  EFI_STATUS Status;
  UINT8 DevBuf[18];
  EFI_USB_DEVICE_REQUEST Req;
  UINT32 tr = 0;
  UINTN Len;

  // GET_DEVICE_DESCRIPTOR (18 bytes)
  ZeroMem(&Req, sizeof(Req));
  Req.RequestType = 0x80; // device-to-host, standard, recipient=device
  Req.Request = USB_REQ_GET_DESCRIPTOR;
  Req.Value = (UINT16)((USB_DESC_TYPE_DEVICE << 8) | 0);
  Req.Index = 0;
  Req.Length = sizeof(DevBuf);
  Len = sizeof(DevBuf);

  Status = CallControlTransfer(Usb2Hc, DeviceAddress, DeviceSpeed, MaxPacketLen, &Req, EfiUsbDataIn, DevBuf, &Len, 500, &tr);
  if (EFI_ERROR(Status) || Len < 8) {
    Print(L"%*sDevice (addr %u): GET_DEVICE_DESCRIPTOR failed (%r)\n", (int)(Depth * 2), L"", (unsigned)DeviceAddress, Status);
    return Status;
  }

  // Parse descriptor bytes (offsets from USB spec)
  UINT8 bDeviceClass = DevBuf[4];
  UINT8 bMaxPacketSize0 = DevBuf[7];
  UINT16 idVendor = (UINT16)(DevBuf[8] | (DevBuf[9] << 8));
  UINT16 idProduct = (UINT16)(DevBuf[10] | (DevBuf[11] << 8));
  Print(L"%*sDevice (addr %u): class=0x%02x vid=0x%04x pid=0x%04x mp0=%u\n",
        (int)(Depth * 2), L"", (unsigned)DeviceAddress, bDeviceClass, idVendor, idProduct, (unsigned)bMaxPacketSize0);

  // Read config header to get total length
  UINT8 confHead[9];
  ZeroMem(&Req, sizeof(Req));
  Req.RequestType = 0x80;
  Req.Request = USB_REQ_GET_DESCRIPTOR;
  Req.Value = (UINT16)((USB_DESC_TYPE_CONFIG << 8) | 0);
  Req.Index = 0;
  Req.Length = sizeof(confHead);
  Len = sizeof(confHead);
  Status = CallControlTransfer(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, &Req, EfiUsbDataIn, confHead, &Len, 500, &tr);
  if (!EFI_ERROR(Status) && Len >= 4) {
    UINT16 totalLen = (UINT16)(confHead[2] | (confHead[3] << 8));
    if (totalLen > 0 && totalLen <= 4096) {
      UINT8 *fullConf = AllocatePool(totalLen);
      if (fullConf) {
        ZeroMem(&Req, sizeof(Req));
        Req.RequestType = 0x80;
        Req.Request = USB_REQ_GET_DESCRIPTOR;
        Req.Value = (UINT16)((USB_DESC_TYPE_CONFIG << 8) | 0);
        Req.Index = 0;
        Req.Length = (UINT16)totalLen;
        Len = totalLen;
        Status = CallControlTransfer(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, &Req, EfiUsbDataIn, fullConf, &Len, 1000, &tr);
        if (!EFI_ERROR(Status) && Len >= 9) {
          UINT8 cfgValue = fullConf[5]; // bConfigurationValue at offset 5
          ZeroMem(&Req, sizeof(Req));
          Req.RequestType = 0x00;
          Req.Request = USB_REQ_SET_CONFIGURATION;
          Req.Value = cfgValue;
          Req.Index = 0;
          Req.Length = 0;
          UINTN z = 0;
          Status = CallControlTransfer(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, &Req, EfiUsbNoData, NULL, &z, 500, &tr);
          if (!EFI_ERROR(Status)) {
            Print(L"%*sDevice (addr %u): SET_CONFIGURATION(%u) OK\n", (int)(Depth * 2), L"", (unsigned)DeviceAddress, (unsigned)cfgValue);
          } else {
            Print(L"%*sDevice (addr %u): SET_CONFIGURATION(%u) failed %r\n", (int)(Depth * 2), L"", (unsigned)DeviceAddress, (unsigned)cfgValue, Status);
          }
        }
        FreePool(fullConf);
      }
	  
	  
	  /*
	  // test here - first version for diagnose 
	  // After you successfully read 'fullConf' (length = totalLen) and possibly called SET_CONFIGURATION
		// Insert this block to parse interfaces and detect hub:
		BOOLEAN isHub = FALSE;
		if (fullConf != NULL && totalLen >= 9) {
		  Print(L"%*s  Parsing configuration descriptors (totalLen=%u)\n", (int)(Depth * 2), L"", (unsigned)totalLen);
		  UINTN pos = 0;
		  while (pos + 1 < totalLen) {
			UINT8 dlen = fullConf[pos];
			UINT8 dtype = fullConf[pos + 1];
			if (dlen == 0) break; // defensive
			if (pos + dlen > totalLen) break; // malformed -> stop
			if (dtype == 4 && dlen >= 9) { // Interface descriptor
			  UINT8 bInterfaceNumber   = fullConf[pos + 2];
			  UINT8 bAlternateSetting  = fullConf[pos + 3];
			  UINT8 bNumEndpoints      = fullConf[pos + 4];
			  UINT8 bInterfaceClass    = fullConf[pos + 5];
			  UINT8 bInterfaceSubClass = fullConf[pos + 6];
			  UINT8 bInterfaceProto    = fullConf[pos + 7];
			  Print(L"%*s    Interface %u alt %u: class=0x%02x subclass=0x%02x proto=0x%02x endpoints=%u\n",
					(int)(Depth * 2), L"",
					(unsigned)bInterfaceNumber,
					(unsigned)bAlternateSetting,
					(unsigned)bInterfaceClass,
					(unsigned)bInterfaceSubClass,
					(unsigned)bInterfaceProto,
					(unsigned)bNumEndpoints);
			  if (bInterfaceClass == HUB_CLASS) {
				isHub = TRUE;
			  }
			}
			pos += dlen;
		  }

		  if (isHub) {
			Print(L"%*s  Device (addr %u) appears to have HUB interface(s). Descend into hub handling.\n",
				  (int)(Depth * 2), L"", (unsigned)DeviceAddress);
			(VOID)EnumerateHubPorts(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, Depth, NextUsbAddr);
		  } else {
			Print(L"%*s  Device (addr %u) does not appear to be a hub (no interface class 0x09 found).\n",
				  (int)(Depth * 2), L"", (unsigned)DeviceAddress);
		  }
		}
		*/

	/*
	ood — that result tells us the device on root port 1 is **not** a hub (so there are no children under it). Below I’ll:

1. Explain exactly what that means and why `GET_HUB_DESCRIPTOR` failed.
2. Give you a small, copy-pasteable diagnostic replacement for your descriptor-parsing block that prints *every* descriptor (hex dump) and more detailed interface/IAD/endpoint info — this will help you identify composite devices, alternate settings, and whether any interface *is* a hub.
3. Offer practical next steps (how to test deeper trees, what to expect).

Short answer: your helper is right — the device reports `bDeviceClass == 0x00` and the configuration descriptors contain no `bInterfaceClass == 0x09` entries, so it is not a hub. `GET_HUB_DESCRIPTOR` therefore rightfully failed (non-hub devices won’t respond to hub-class requests).
*/


	  //
// Enhanced debug/parse of fullConf (call this after successful GET_CONFIGURATION full read)
// fullConf: UINT8* buffer, totalLen: UINT16 or UINTN length
//
{
  BOOLEAN isHub = FALSE;

  Print(L"%*sParsing configuration (totalLen=%u)\n", (int)(Depth * 2), L"", (unsigned)totalLen);

  // Hexdump (compact) - show up to first 256 bytes to avoid huge logs
  {
    UINTN show = totalLen;
    if (show > 256) show = 256;
    Print(L"%*sConfig hex dump (first %u bytes):\n", (int)(Depth * 2), L"", (unsigned)show);
    for (UINTN i = 0; i < show; i += 16) {
      Print(L"%*s%04x: ", (int)(Depth * 2), L"", (unsigned)i);
      for (UINTN j = 0; j < 16 && i + j < show; ++j) {
        Print(L"%02x ", (unsigned)fullConf[i + j]);
      }
      Print(L"\n");
    }
    if (totalLen > show) {
      Print(L"%*s... (total %u bytes)\n", (int)(Depth * 2), L"", (unsigned)totalLen);
    }
  }

  // Walk descriptors
  UINTN pos = 0;
  while (pos + 1 < totalLen) {
    UINT8 dlen = fullConf[pos];
    UINT8 dtype = fullConf[pos + 1];
    if (dlen == 0) {
      Print(L"%*sDescriptor at pos %u has length 0 - stopping parse\n", (int)(Depth * 2), L"", (unsigned)pos);
      break;
    }
    if (pos + dlen > totalLen) {
      Print(L"%*sMalformed descriptor at pos %u: length %u exceeds total %u - stopping\n",
            (int)(Depth * 2), L"", (unsigned)pos, (unsigned)dlen, (unsigned)totalLen);
      break;
    }

    switch (dtype) {
      case 0x01: // Device (shouldn't be inside config but print if found)
        Print(L"%*s[Device Desc] len=%u\n", (int)(Depth * 2), L"", (unsigned)dlen);
        break;
      case 0x02: // Configuration
        if (dlen >= 9) {
          UINT16 wTotalLength = (UINT16)(fullConf[pos+2] | (fullConf[pos+3] << 8));
          UINT8 bNumInterfaces = fullConf[pos+4];
          UINT8 bConfigurationValue = fullConf[pos+5];
          Print(L"%*s[Config Desc] len=%u total=%u interfaces=%u cfgValue=%u\n",
                (int)(Depth * 2), L"", (unsigned)dlen, (unsigned)wTotalLength, (unsigned)bNumInterfaces, (unsigned)bConfigurationValue);
        } else {
          Print(L"%*s[Config Desc] len=%u\n", (int)(Depth * 2), L"", (unsigned)dlen);
        }
        break;
      case 0x04: // Interface
        if (dlen >= 9) {
          UINT8 bInterfaceNumber = fullConf[pos+2];
          UINT8 bAlternateSetting  = fullConf[pos+3];
          UINT8 bNumEndpoints      = fullConf[pos+4];
          UINT8 bInterfaceClass    = fullConf[pos+5];
          UINT8 bInterfaceSubClass = fullConf[pos+6];
          UINT8 bInterfaceProto    = fullConf[pos+7];
          Print(L"%*s[Interface] #%u alt=%u endpoints=%u class=0x%02x sub=0x%02x proto=0x%02x\n",
                (int)(Depth * 2), L"", (unsigned)bInterfaceNumber, (unsigned)bAlternateSetting, (unsigned)bNumEndpoints,
                (unsigned)bInterfaceClass, (unsigned)bInterfaceSubClass, (unsigned)bInterfaceProto);
          if (bInterfaceClass == 0x09) {
            isHub = TRUE;
            Print(L"%*s  -> Interface indicates HUB class (0x09)\n", (int)(Depth * 2), L"");
          }
        } else {
          Print(L"%*s[Interface] len=%u\n", (int)(Depth * 2), L"", (unsigned)dlen);
        }
        break;
      case 0x0B: // Interface Association Descriptor (IAD)
        if (dlen >= 8) {
          UINT8 firstIf = fullConf[pos+2];
          UINT8 lastIf  = fullConf[pos+3];
          UINT8 funcClass = fullConf[pos+4];
          Print(L"%*s[IAD] first=%u last=%u funcClass=0x%02x\n", (int)(Depth * 2), L"", (unsigned)firstIf, (unsigned)lastIf, (unsigned)funcClass);
        } else {
          Print(L"%*s[IAD] len=%u\n", (int)(Depth * 2), L"", (unsigned)dlen);
        }
        break;
      case 0x05: // Endpoint
        if (dlen >= 7) {
          UINT8 bEndpointAddr = fullConf[pos+2];
          UINT8 bmAttributes = fullConf[pos+3];
          UINT16 wMaxPacket = (UINT16)(fullConf[pos+4] | (fullConf[pos+5] << 8));
          Print(L"%*s[Endpoint] addr=0x%02x attr=0x%02x maxpktsz=%u\n", (int)(Depth * 2), L"", (unsigned)bEndpointAddr, (unsigned)bmAttributes, (unsigned)wMaxPacket);
        } else {
          Print(L"%*s[Endpoint] len=%u\n", (int)(Depth * 2), L"", (unsigned)dlen);
        }
        break;
      case 0x29: // Hub descriptor (class-specific, rarely in config)
        Print(L"%*s[Hub desc found inside config?] len=%u\n", (int)(Depth * 2), L"", (unsigned)dlen);
        break;
      default:
        Print(L"%*s[Desc type 0x%02x] len=%u\n", (int)(Depth * 2), L"", (unsigned)dtype, (unsigned)dlen);
        break;
    }

    pos += dlen;
  }

  if (isHub) {
    Print(L"%*sDevice (addr %u) has interface class 0x09 -> treat as hub and call EnumerateHubPorts()\n",
          (int)(Depth * 2), L"", (unsigned)DeviceAddress);
    (VOID)EnumerateHubPorts(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, Depth, NextUsbAddr);
  } else {
    Print(L"%*sDevice (addr %u) not a hub (no interface class 0x09 found)\n",
          (int)(Depth * 2), L"", (unsigned)DeviceAddress);
  }
}


/*

  Parsing configuration (totalLen=32)
  Config hex dump (first 32 bytes):
  0000: 00 98 00 00 00 00 00 00 D0 00 FD 00 00 00 00 00
  0010: 00 74 00 6C 01 02 00 02 70 00 61 00 00 00 00 00
  Malformed descriptor at pos 0: length 152 exceeds total 32 - stopping
  Device (addr 1) not a hub (no interface class 0x09 found)
  
  -----------------
  
  Short diagnosis: your full-configuration buffer (fullConf) doesn’t contain a valid USB configuration descriptor stream — the first byte is 0x00 (descriptor length 0), so the parser immediately stops. The hex dump shows many 0x00 bytes interleaved (00 98 00 00 ... 70 00 61 00 ...) which looks like UTF-16LE (wide chars) or otherwise 16-bit-aligned data — not a normal USB descriptor byte stream.

That can happen for a few reasons:

The full GET_DESCRIPTOR transfer returned far fewer bytes than the wTotalLength you read from the 9-byte header (you tried to parse using the larger wTotalLength = 0x0098 (=152) but only 32 bytes were returned — hence the “length 152 exceeds total 32” message). Always use the actual returned length when parsing.

The returned data is not what you requested (bmRequestType, Request, Value, Index, or DeviceAddress wrong), so the device returned an unexpected descriptor or some other data (or the controller copied the wrong buffer).

The buffer you passed to the transfer was the wrong pointer / type or later overwritten / printed as the wrong element-size (e.g., treated as CHAR16* or memory became populated with UTF-16 text). That interleaved 00 pattern strongly suggests the buffer is being treated/printed as bytes but it contains UTF-16 text (or you accidentally printed a CHAR16 buffer as bytes).

Below I give a robust replacement for the GET CONFIG + parse flow. It:

Reads the 9-byte configuration header and prints it (hex).

Computes wTotalLength from the header.

Requests the full configuration with wTotalLength but uses the actual returned length for parsing.

Zeroes the buffer, prints the actual returned length & TransferResult for diagnostics.

If the returned data looks suspicious (first byte 0 or many zero bytes), it prints a note and a 16-bit view to help spot UTF-16 accidental-copy bugs.

Parses descriptors using the actual returned length (ReturnedLen) not the header wTotalLength, and only calls hub handling when an interface class==0x09 is really found.

Replace your current full-config read / parse block with the code below inside EnumerateDeviceRecursive() (use the same variable names or adapt — it expects Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, Depth, NextUsbAddr and that you already read the device descriptor earlier):
*/


/* --- robust GET_CONFIGURATION + diagnostic parse --- */
/*
{
  //EFI_USB_DEVICE_REQUEST Req;
  UINTN ReturnedLen;
  UINT32 TransferResult;
  //UINTN Status;

  // 1) Read first 9 bytes of configuration descriptor (header)
  //UINT8 confHead[9];
  ZeroMem(&Req, sizeof(Req));
  Req.RequestType = 0x80; // device-to-host, standard, recipient=device
  Req.Request     = USB_REQ_GET_DESCRIPTOR;
  Req.Value       = (UINT16)((USB_DESC_TYPE_CONFIG << 8) | 0);
  Req.Index       = 0;
  Req.Length      = sizeof(confHead);
  ReturnedLen = sizeof(confHead);

  Status = CallControlTransfer(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, &Req, EfiUsbDataIn, confHead, &ReturnedLen, 1000, &TransferResult);
  if (EFI_ERROR(Status) || ReturnedLen < 4) {
    Print(L"%*sGET_CONFIG_HEADER failed (Status=%r, Returned=%u, tr=0x%x)\n", (int)(Depth*2), L"", Status, (unsigned)ReturnedLen, TransferResult);
    // bail or continue conservatively
  } else {
    // Print header bytes for debugging
    Print(L"%*sConfig header (returned %u bytes):", (int)(Depth*2), L"", (unsigned)ReturnedLen);
    for (UINTN i = 0; i < ReturnedLen; ++i) { Print(L" %02x", (unsigned)confHead[i]); }
    Print(L"\n");

    // Interpret wTotalLength from header (little-endian)
    UINT16 wTotalLength = (UINT16)(confHead[2] | (confHead[3] << 8));
    Print(L"%*sReported wTotalLength = %u\n", (int)(Depth*2), L"", (unsigned)wTotalLength);

    if (wTotalLength == 0 || wTotalLength > 65535) {
      Print(L"%*sSuspicious total length %u - aborting full-conf read\n", (int)(Depth*2), L"", (unsigned)wTotalLength);
    } else {
      // 2) Allocate buffer and request full configuration (request up to wTotalLength)
      //UINT8 *fullConf = AllocateZeroPool(wTotalLength);
      if (fullConf == NULL) {
        Print(L"%*sFailed to allocate %u bytes for fullConf\n", (int)(Depth*2), L"", (unsigned)wTotalLength);
      } else {
        ZeroMem(&Req, sizeof(Req));
        Req.RequestType = 0x80;
        Req.Request     = USB_REQ_GET_DESCRIPTOR;
        Req.Value       = (UINT16)((USB_DESC_TYPE_CONFIG << 8) | 0);
        Req.Index       = 0;
        Req.Length      = (UINT16)wTotalLength;
        ReturnedLen = wTotalLength;

        Status = CallControlTransfer(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, &Req, EfiUsbDataIn, fullConf, &ReturnedLen, 3000, &TransferResult);
        Print(L"%*sGET_CONFIG full -> Status=%r Returned=%u tr=0x%x\n", (int)(Depth*2), L"", Status, (unsigned)ReturnedLen, TransferResult);

        if (EFI_ERROR(Status) || ReturnedLen < 9) {
          Print(L"%*sGET_CONFIG full failed or too short (len=%u). Not parsing.\n", (int)(Depth*2), L"", (unsigned)ReturnedLen);
        } else {
          // Diagnostic hex dump (first up to 256 bytes)
          UINTN show = ReturnedLen;
          if (show > 256) show = 256;
          Print(L"%*sFull config dump (first %u bytes):\n", (int)(Depth*2), L"", (unsigned)show);
          for (UINTN i = 0; i < show; i += 16) {
            Print(L"%*s%04x: ", (int)(Depth*2), L"", (unsigned)i);
            for (UINTN j = 0; j < 16 && i + j < show; ++j) {
              Print(L"%02x ", (unsigned)fullConf[i + j]);
            }
            Print(L"\n");
          }

          // If the first byte is 0, print a 16-bit view to expose UTF-16-like data
          if (fullConf[0] == 0) {
            Print(L"%*sNote: first byte is 0x00 — buffer looks 16-bit/UTF-16LE-like. Printing as 16-bit words:\n", (int)(Depth*2), L"");
            for (UINTN i = 0; i + 1 < show; i += 2) {
              UINT16 w = (UINT16)(fullConf[i] | (fullConf[i+1] << 8));
              Print(L"%04x ", (unsigned)w);
              if (((i/2) & 0xF) == 0xF) Print(L"\n");
            }
            Print(L"\n");
          }

          // Parse descriptors using ReturnedLen (not the reported wTotalLength)
          UINTN pos = 0;
          BOOLEAN isHub = FALSE;
          while (pos + 1 < ReturnedLen) {
            UINT8 dlen = fullConf[pos];
            UINT8 dtype = fullConf[pos + 1];
            if (dlen == 0) {
              Print(L"%*sDescriptor at pos %u has length 0 - stop\n", (int)(Depth*2), L"", (unsigned)pos);
              break;
            }
            if (pos + dlen > ReturnedLen) {
              Print(L"%*sDescriptor at pos %u overruns returned length (dlen=%u returned=%u) - stop\n",
                    (int)(Depth*2), L"", (unsigned)pos, (unsigned)dlen, (unsigned)ReturnedLen);
              break;
            }

            if (dtype == 4 && dlen >= 9) {
              UINT8 bInterfaceClass = fullConf[pos+5];
              Print(L"%*sFound Interface desc at pos %u: class=0x%02x\n", (int)(Depth*2), L"", (unsigned)pos, (unsigned)bInterfaceClass);
              if (bInterfaceClass == 0x09) {
                isHub = TRUE;
              }
            }

            pos += dlen;
          } // parse loop

          if (isHub) {
            Print(L"%*sDevice (addr %u) appears to be a hub -> EnumerateHubPorts()\n", (int)(Depth*2), L"", (unsigned)DeviceAddress);
            (VOID)EnumerateHubPorts(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, Depth, NextUsbAddr);
          } else {
            Print(L"%*sDevice (addr %u) not a hub (no interface class 0x09 found)\n", (int)(Depth*2), L"", (unsigned)DeviceAddress);
          }
        } // fullConf returned OK

        FreePool(fullConf);
      } // allocated fullConf
    } // wTotalLength ok
  } // confHead ok
} // end block

*/
	  
	  
	  
	  
    }
  }

  // If device class suggests hub, attempt hub handling (EnumerateHubPorts will attempt GET_HUB_DESCRIPTOR)
  if (bDeviceClass == HUB_CLASS || bDeviceClass == 0) {
    (void)EnumerateHubPorts(Usb2Hc, DeviceAddress, DeviceSpeed, bMaxPacketSize0, Depth, NextUsbAddr);
  }

  return EFI_SUCCESS;
}


//
// Minimal EHCI helper — best-effort introspect + try to use EFI_USB2_HC_PROTOCOL
//
// This reads capability registers, computes operational base,
// prints USBCMD/USBSTS and per-port PORTSC values, then tries to
// find the EFI_USB2_HC_PROTOCOL instance and invoke EnumerateRootHub().
//
// References: EHCI spec (CAPLENGTH, HCSPARAMS, PORTSC offset). :contentReference[oaicite:0]{index=0}
//
STATIC
EFI_STATUS
InitializeEhciControllerBeta (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN EFI_HANDLE          PciHandle
  )
{
  EFI_STATUS Status;
  UINT8  BarIndex = 0; // assume MMIO in BAR0
  UINT32 cap0;
  UINT8  capLength;
  UINT32 hcsp;
  UINT32 numPorts;
  UINT32 opOffset; // offset from BAR0 to operational regs (caplength)
  UINT32 usbcmd;
  UINT32 usbsts;
  UINT32 portVal;
  UINT32 i;

  Print(L"  Detected EHCI host controller. Running minimal introspect...\n");

  // Read first 32-bit of capability registers (contains CAPLENGTH in low 8 bits)
  Status = OhciRead32(PciIo, BarIndex, 0x00, &cap0);
  if (EFI_ERROR(Status)) {
    Print(L"    EHCI: failed to read CAP (offset 0) (%r)\n", Status);
    return Status;
  }
  capLength = (UINT8)(cap0 & 0xFF);
  Print(L"    CAP (DWORD @0) = 0x%08x  CAPLENGTH = %u\n", cap0, (unsigned)capLength);

  // Read HCSPARAMS (HCSPARAMS at offset 0x04 in capability space)
  Status = OhciRead32(PciIo, BarIndex, 0x04, &hcsp);
  if (EFI_ERROR(Status)) {
    Print(L"    EHCI: failed to read HCSPARAMS (offset 0x04) (%r)\n", Status);
    return Status;
  }

  // Number of ports is encoded in HCSPARAMS.  Per EHCI spec / kernel macros,
  // N_PORTS is in bits [3:0] of HCSPARAMS. (see EHCI spec / kernel headers). :contentReference[oaicite:1]{index=1}
  numPorts = (hcsp & 0x0F);
  Print(L"    HCSPARAMS = 0x%08x  -> N_PORTS = %u\n", hcsp, (unsigned)numPorts);

  // Compute operational registers base: BAR + CAPLENGTH (CAPLENGTH is bytes)
  // We will read op registers as BAR0 + capLength + offset.
  opOffset = (UINT32)capLength; // add this to BAR base when addressing op regs
  Print(L"    Operational registers base offset (caplen) = 0x%08x\n", opOffset);

  // Read a couple of operational registers for debug.
  // USBCMD at opBase + 0x00, USBSTS at opBase + 0x04 per EHCI spec. :contentReference[oaicite:2]{index=2}
  Status = OhciRead32(PciIo, BarIndex, opOffset + 0x00, &usbcmd);
  if (EFI_ERROR(Status)) {
    Print(L"    EHCI: failed to read USBCMD (%r)\n", Status);
  } else {
    Print(L"    USBCMD = 0x%08x\n", usbcmd);
  }

  Status = OhciRead32(PciIo, BarIndex, opOffset + 0x04, &usbsts);
  if (EFI_ERROR(Status)) {
    Print(L"    EHCI: failed to read USBSTS (%r)\n", Status);
  } else {
    Print(L"    USBSTS = 0x%08x\n", usbsts);
  }

  // Read port status/control registers.  PORTSC registers start at opBase + 0x44,
  // each port is 32-bit wide and consecutive. (EHCI operational registers layout). :contentReference[oaicite:3]{index=3}
  if (numPorts > 0 && numPorts < 32) {
    UINT32 portBaseOffset = opOffset + 0x44;
    for (i = 0; i < numPorts; ++i) {
      UINT32 portOffset = portBaseOffset + (i * 4);
      Status = OhciRead32(PciIo, BarIndex, portOffset, &portVal);
      if (EFI_ERROR(Status)) {
        Print(L"    Port %u: read PORTSC@0x%08x failed (%r)\n", (unsigned)(i + 1), portOffset, Status);
        continue;
      }
      Print(L"    Port %u: PORTSC=0x%08x\n", (unsigned)(i + 1), portVal);
    }
  } else {
    Print(L"    EHCI: unusual port count %u, skipping port reads\n", (unsigned)numPorts);
  }

  //
  // Now try to find an EFI_USB2_HC_PROTOCOL published by the host controller driver.
  // First attempt directly on the PCI handle (may be absent), then locate matching HC handle.
  //
  EFI_USB2_HC_PROTOCOL *Usb2Hc = NULL;
  Status = gBS->HandleProtocol(PciHandle, &gEfiUsb2HcProtocolGuid, (VOID**)&Usb2Hc);
  if (EFI_ERROR(Status) || Usb2Hc == NULL) {
    Print(L"    EFI_USB2_HC_PROTOCOL not found on this PCI handle; searching system for matching HC handle...\n");

    // Try to find the HC handle that corresponds to this PCI device (by comparing vendor/device/class)
    EFI_HANDLE *foundHandlePtr = NULL;
    EFI_HANDLE foundHcHandle = NULL;
    EFI_USB2_HC_PROTOCOL *foundUsb2 = NULL;

    // Attempt to reuse FindUsb2HcForPciHandle if you added it earlier in the file.
    // If not present, fall back to locating any handle with EFI_USB2_HC_PROTOCOL.
    #ifdef HAS_FindUsb2HcForPciHandle
    {
      EFI_STATUS s = FindUsb2HcForPciHandle(PciHandle, PciIo, &foundHandlePtr, &foundUsb2);
      if (!EFI_ERROR(s) && foundUsb2 != NULL) {
        Usb2Hc = foundUsb2;
        foundHcHandle = (foundHandlePtr ? *foundHandlePtr : NULL);
      }
    }
    #endif

    if (Usb2Hc == NULL) {
      // fallback: locate any handles that implement the protocol and pick the first
      EFI_HANDLE *HcHandles = NULL;
      UINTN HcCount = 0;
      Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HcCount, &HcHandles);
      if (EFI_ERROR(Status) || HcCount == 0) {
        Print(L"    No EFI_USB2_HC_PROTOCOL instances found in system - cannot perform high-level enumeration.\n");
      } else {
        // Try to open first one
        Status = gBS->HandleProtocol(HcHandles[0], &gEfiUsb2HcProtocolGuid, (VOID**)&Usb2Hc);
        if (EFI_ERROR(Status) || Usb2Hc == NULL) {
          Print(L"    Found HC handle but opening EFI_USB2_HC_PROTOCOL failed: %r\n", Status);
        } else {
          foundHcHandle = HcHandles[0];
          Print(L"    Using EFI_USB2_HC_PROTOCOL from handle %p (protocol at %p)\n", foundHcHandle, Usb2Hc);
        }
        FreePool(HcHandles);
      }
    } else {
      Print(L"    Found matching HC handle %p (USB2_HC at %p)\n", (VOID*)foundHcHandle, foundUsb2);
    }

    if (foundHandlePtr) {
      // caller-provided allocation — free it if allocated
      FreePool(foundHandlePtr);
    }
  } else {
    Print(L"    EFI_USB2_HC_PROTOCOL present on this handle (protocol at %p)\n", Usb2Hc);
  }

  // If we got a USB2_HC protocol instance, call the higher-level enumerator you already have.
  if (Usb2Hc != NULL) {
    UINT8 nextAddr = 1;
    Print(L"    Launching high-level enumeration via EFI_USB2_HC_PROTOCOL (NumPorts=%u)\n", (unsigned)numPorts);
    (VOID)EnumerateRootHub(Usb2Hc, (UINT32)numPorts, &nextAddr);
  } else {
    Print(L"    No usable EFI_USB2_HC_PROTOCOL for this EHCI controller; skipping high-level enumeration.\n");
  }

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
  
  // setup GOP
	Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&mGraphicsOuput);
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	UINTN gop_querymode_size = sizeof(EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
	EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *mode_info = NULL;
	Status = mGraphicsOuput->QueryMode(mGraphicsOuput, mGraphicsOuput->Mode->Mode, 
		&gop_querymode_size, &mode_info); 

	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);

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
	  
	  // update 13-10-2025
	  // In default demo this part doesen't exists, this is extra stuff for real HW to test 0x20 ProgIf also
	  // if OHCI is not detected, its failed then go to next round
	  // second round for EHCI 0x20
	  if (ProgIf == PCI_PROGIF_EHCI) {
        InitializeEhciControllerBeta(PciIo, PciHandles[idx]); // do the same stuff but for EHCI hub and recursively find childrens
      } else {
		Print(L" ( Ehci examine error ) \n" );
	  }

		// last print
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
