/** UsbDiagApp.c
  UEFI app that diagnoses USB stack step-by-step and prints results to console.
  Based on HelloWorld sample.

  SPDX-License-Identifier: BSD-2-Clause-Patent
**/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/PrintLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Protocol/PciIo.h>
#include <Protocol/UsbIo.h>
#include <Protocol/Usb2HostController.h>
#include <Protocol/UsbHostController.h>
#include <Protocol/BlockIo.h>
#include <Guid/GlobalVariable.h>
#include <IndustryStandard/Usb.h>

#define TIMEOUT_MS 5000
#define MAX_BARS 6
#define MAX_PORTS_TO_PRINT 16

STATIC
VOID
DumpHex (
  IN CHAR16   *Label,
  IN UINT8    *Buf,
  IN UINTN    Len
  )
{
  UINTN i;
  if (Label != NULL) {
    Print (L"%s (len=%u):\n", Label, (UINT32)Len);
  }
  for (i = 0; i < Len; i++) {
    if ((i % 16) == 0) {
      Print (L"%04x: ", (UINT32)i);
    }
    Print (L"%02x ", Buf[i]);
    if ((i % 16) == 15) {
      Print (L"\n");
    }
  }
  if ((Len % 16) != 0) {
    Print (L"\n");
  }
}

STATIC
VOID
PrintStatus (
  IN CHAR16    *Prefix,
  IN EFI_STATUS Status
  )
{
  Print (L"%s: Status = 0x%lx\n", Prefix, Status);
}

/**
  Scan all handles that expose EFI_PCI_IO_PROTOCOL and print which are USB controllers.
**/
STATIC
VOID
ScanPciForUsbControllers (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN Index;

  Print (L"--- Stage 1: Scan PCI bus for USB class devices ---\n");

  Status = gBS->LocateHandleBuffer (
                  ByProtocol,
                  &gEfiPciIoProtocolGuid,
                  NULL,
                  &HandleCount,
                  &Handles
                  );
  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(PciIo) failed: 0x%lx\n", Status);
    return;
  }

  for (Index = 0; Index < HandleCount; ++Index) {
    EFI_PCI_IO_PROTOCOL *PciIo;
    UINT32 Data32 = 0;
    UINTN Bus, Device, Function, Segment;
    Status = gBS->HandleProtocol (Handles[Index], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR (Status)) {
      Print (L"HandleProtocol(PciIo) failed for handle %u: 0x%lx\n", Index, Status);
      continue;
    }

    // Get the B:D:F
    PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);

    // Read DWORD at offset 0x08 which contains RevisionID | ProgIF | Subclass | Class
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &Data32);
    if (EFI_ERROR (Status)) {
      Print (L" Pci.Read failed for %02x:%02x.%x : 0x%lx\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function, Status);
      continue;
    }

    UINT8 ClassCode = (UINT8)((Data32 >> 24) & 0xFF);
    UINT8 SubClass  = (UINT8)((Data32 >> 16) & 0xFF);
    UINT8 ProgIf    = (UINT8)((Data32 >> 8) & 0xFF);

    Print (L"PCI %02x:%02x.%x  Class=0x%02x SubClass=0x%02x ProgIf=0x%02x\n",
           (UINT32)Bus, (UINT32)Device, (UINT32)Function, ClassCode, SubClass, ProgIf);

    if (ClassCode == 0x0C && SubClass == 0x03) {
      Print (L"  -> Detected USB controller (PCI class 0x0C/0x03). Try to locate HC protocols on same handle.\n");
    }
  }

  if (Handles != NULL) {
    FreePool (Handles);
  }
}

/**
  Locate Host Controller protocol handles and print counts.
**/
STATIC
VOID
FindUsbHostControllerProtocols (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;

  Print (L"--- Stage 2: Find published USB Host Controller protocols ---\n");

  // Try USB2 HC first
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &Count, &Handles);
  if (!EFI_ERROR (Status) && Count > 0) {
    Print (L"Found %u USB2_HC protocol handle(s)\n", (UINT32)Count);
    FreePool (Handles);
  } else {
    Print (L"No USB2_HC handles found (Status=0x%lx)\n", Status);
  }

  // Try legacy USB_HC protocol (USB 1.1)
  Handles = NULL;
  Count = 0;
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsbHcProtocolGuid, NULL, &Count, &Handles);
  if (!EFI_ERROR (Status) && Count > 0) {
    Print (L"Found %u USB_HC protocol handle(s)\n", (UINT32)Count);
    FreePool (Handles);
  } else {
    Print (L"No USB_HC handles found (Status=0x%lx)\n", Status);
  }
}

/**
  Map EFI_USB2_HC_PROTOCOL handles to PCI B:D.F (if PciIo available on same handle)
**/
STATIC
VOID
MapUsb2HcToPciBdf (VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *HcHandles = NULL;
  UINTN HcCount = 0;
  UINTN i;

  Print (L"--- Mapping USB2_HC handles to PCI B:D.F ---\n");
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HcCount, &HcHandles);
  if (EFI_ERROR (Status) || HcCount == 0) {
    Print (L"  No USB2_HC handles: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HcCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    UINTN Bus, Device, Function, Segment;

    Status = gBS->HandleProtocol (HcHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (!EFI_ERROR (Status) && PciIo != NULL) {
      PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);
      Print (L"  USB2_HC handle %u -> PCI %02x:%02x.%x (Segment=%u)\n", (UINT32)i, (UINT32)Bus, (UINT32)Device, (UINT32)Function, (UINT32)Segment);
    } else {
      Print (L"  USB2_HC handle %u -> PciIo not on same handle (or not accessible): 0x%lx\n", (UINT32)i, Status);
    }
  }

  if (HcHandles) {
    FreePool (HcHandles);
  }
}

/**
  Probe PCI handle BARs for EHCI capability + operational registers and print key registers.
  Read-only; does not write registers.
**/
STATIC
VOID
CheckEhciRegistersForPciHandle (
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;
  EFI_PCI_IO_PROTOCOL *PciIo;
  UINT32 Data32;
  UINTN Bus, Device, Function, Segment;
  //UINTN BarIndex;
  UINT8 BarIndex;

  Status = gBS->HandleProtocol (Handle, &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
  if (EFI_ERROR (Status)) {
    return;
  }

  PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);

  // Read DWORD at 0x08 (Class/Subclass/ProgIf/RevID)
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &Data32);
  if (EFI_ERROR (Status)) {
    return;
  }
  UINT8 ClassCode = (UINT8)((Data32 >> 24) & 0xFF);
  UINT8 SubClass  = (UINT8)((Data32 >> 16) & 0xFF);
  UINT8 ProgIf    = (UINT8)((Data32 >> 8) & 0xFF);

  if (!(ClassCode == 0x0C && SubClass == 0x03)) {
    // Not a USB controller
    return;
  }

  Print (L"--- EHCI probe for PCI %02x:%02x.%x | PROGIF %02x ---\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function, (UINT32)ProgIf);

  for (BarIndex = 0; BarIndex < MAX_BARS; ++BarIndex) {
    UINT8 CapLength = 0xFF;
    // Try reading the first byte of BAR region (cap length)
    Status = PciIo->Mem.Read (
                       PciIo,
                       EfiPciIoWidthUint8,
                       BarIndex,
                       0x00,
                       1,
                       &CapLength
                       );
    if (EFI_ERROR (Status)) {
      continue;
    }

    if (CapLength == 0xFF || CapLength == 0x00) {
      continue;
    }

    UINT16 HciVersion = 0;
    Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint16, BarIndex, 0x02, 1, &HciVersion);
    if (EFI_ERROR (Status)) {
      continue;
    }

    Print (L"  BAR%u: CAPLENGTH=0x%02x HCIVERSION=0x%04x\n", (UINT32)BarIndex, CapLength, HciVersion);

    UINTN OpBase = (UINTN)CapLength;

    #define READ_OP32(off, var) \
      do { \
        EFI_STATUS _st = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, (off) + OpBase, 1, &(var)); \
        if (EFI_ERROR (_st)) { Print (L"    Read op @0x%03x failed: 0x%lx\n", (UINT32)((off)+OpBase), _st); } \
      } while (0)

    UINT32 UsbCmd = 0, UsbSts = 0, UsbIntr = 0, FrIndex = 0, AsyncList = 0, PeriodicBase = 0;
    READ_OP32 (0x00, UsbCmd);       // USBCMD offset 0x00
    READ_OP32 (0x04, UsbSts);       // USBSTS offset 0x04
    READ_OP32 (0x08, UsbIntr);      // USBINTR offset 0x08
    READ_OP32 (0x0C, FrIndex);      // FRINDEX offset 0x0C
    READ_OP32 (0x18, AsyncList);    // ASYNCLISTADDR offset 0x18
    READ_OP32 (0x10, PeriodicBase); // PERIODICLISTBASE offset 0x10

    Print (L"    USBCMD=0x%08x USBSTS=0x%08x USBINTR=0x%08x FRINDEX=0x%08x\n", UsbCmd, UsbSts, UsbIntr, FrIndex);
    Print (L"    PERIODICLISTBASE=0x%08x ASYNCLISTADDR=0x%08x\n", PeriodicBase, AsyncList);

    // Read HCSPARAMS (capability offset 0x04)
    UINT32 HcSParams = 0;
    Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);
    if (!EFI_ERROR (Status)) {
      // display raw and approximate port count (lowest 4 bits often used)
      UINT32 NPorts = (HcSParams & 0x0000000F);
      Print (L"    HCSPARAMS(raw)=0x%08x  (N_PORTS approx=%u)\n", HcSParams, NPorts);
    } else {
      Print (L"    HCSPARAMS read failed: 0x%lx\n", Status);
    }

    // Read PORTSC registers: operational offset 0x44 + 4 * port
    UINTN PortIndex;
    for (PortIndex = 0; PortIndex < MAX_PORTS_TO_PRINT; ++PortIndex) {
      UINT32 PortSc = 0;
      UINTN PortOffset = OpBase + 0x44 + (UINTN)PortIndex * 4;
      EFI_STATUS _r = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
      if (EFI_ERROR (_r)) {
        // stop if reading next port fails
        break;
      }
      Print (L"      PORT%u: PORTSC raw=0x%08x  (CCS=%d CSC=%d PE=%d PortSpeed=%d PO=%d)\n",
             (UINT32)PortIndex,
             PortSc,
             (int)((PortSc >> 0) & 1),         // Current Connect Status
             (int)((PortSc >> 1) & 1),         // Connect Status Change
             (int)((PortSc >> 2) & 1),         // Port Enabled
             (int)((PortSc >> 26) & 3),        // simplified port speed decode (spec varies)
             (int)((PortSc >> 13) & 1)         // Port Owner (example bit position)
             );
    }

    // stop after first viable EHCI region found
    break;
  } // for each BAR
}

/**
  Scan all PCI handles and run the EHCI probe for USB controllers.
**/
STATIC
VOID
ScanAllPciForEhciAndCheck (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN i;

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(PciIo) failed: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HandleCount; ++i) {
    CheckEhciRegistersForPciHandle (Handles[i]);
  }

  if (Handles) {
    FreePool (Handles);
  }
}

/**
  Find BlockIo handles and read LBA0 (safe read) for devices where MediaPresent is true.
**/
STATIC
VOID
CheckUsbBlockIoAndReadLba0 (VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN i;

  Print (L"--- Stage: Check BlockIo on USB mass-storage devices ---\n");

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiBlockIoProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR (Status) || HandleCount == 0) {
    Print (L"  No BlockIo handles found: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HandleCount; ++i) {
    EFI_BLOCK_IO_PROTOCOL *Blk = NULL;
    Status = gBS->HandleProtocol (Handles[i], &gEfiBlockIoProtocolGuid, (VOID**)&Blk);
    if (EFI_ERROR (Status) || Blk == NULL) {
      continue;
    }

    Print (L"  BlockIo handle %u: MediaId=%u BlockSize=%u LastBlock=%llu MediaPresent=%u Removable=%u\n",
           (UINT32)i, (UINT32)Blk->Media->MediaId, (UINT32)Blk->Media->BlockSize,
           Blk->Media->LastBlock, (UINT32)Blk->Media->MediaPresent, (UINT32)Blk->Media->RemovableMedia);

    if (!Blk->Media->MediaPresent) {
      Print (L"    Media not present, skipping read.\n");
      continue;
    }

    UINTN BufSize = (UINTN)Blk->Media->BlockSize;
    VOID *Buf = AllocatePool (BufSize);
    if (Buf == NULL) {
      Print (L"    AllocatePool failed for %u bytes\n", (UINT32)BufSize);
      continue;
    }
    ZeroMem (Buf, BufSize);

    Status = Blk->ReadBlocks (Blk, Blk->Media->MediaId, 0, BufSize, Buf);
    Print (L"    ReadBlocks(LBA0) returned 0x%lx\n", Status);
    if (!EFI_ERROR (Status)) {
      DumpHex (L"    LBA0 (first 64 bytes)", (UINT8*)Buf, (BufSize > 64) ? 64 : BufSize);
      // Check for MBR signature 0x55AA at offset 0x1FE if block is >= 512
      if (BufSize >= 512) {
        UINT8 *b = (UINT8*)Buf;
        if (b[510] == 0x55 && b[511] == 0xAA) {
          Print (L"    MBR signature 0x55AA found at offset 0x1FE\n");
        } else {
          Print (L"    No MBR signature at 0x1FE (bytes=0x%02x 0x%02x)\n", b[510], b[511]);
        }
      }
    }
    FreePool (Buf);
  }

  if (Handles) {
    FreePool (Handles);
  }
}

/**
  Exercise EFI_USB_IO_PROTOCOL methods for each published USB device.
**/
STATIC
VOID
CheckUsbDevices (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN Index;

  Print (L"--- Stage 3: Locate EFI_USB_IO_PROTOCOL handles (actual devices) ---\n");

  Status = gBS->LocateHandleBuffer (
                  ByProtocol,
                  &gEfiUsbIoProtocolGuid,
                  NULL,
                  &HandleCount,
                  &Handles
                  );
  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(UsbIo) failed: 0x%lx\n", Status);
    return;
  }
  Print (L"Found %u USB device handles (EFI_USB_IO_PROTOCOL)\n", (UINT32)HandleCount);

  for (Index = 0; Index < HandleCount; ++Index) {
    EFI_USB_IO_PROTOCOL *UsbIo;
    Print (L"\n--- Device %u ---\n", (UINT32)Index);

    Status = gBS->HandleProtocol (Handles[Index], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
    if (EFI_ERROR (Status) || UsbIo == NULL) {
      Print (L" HandleProtocol(UsbIo) failed: 0x%lx\n", Status);
      continue;
    }

    // 1) Get Device Descriptor
    {
      EFI_USB_DEVICE_DESCRIPTOR DevDesc;
      ZeroMem (&DevDesc, sizeof(DevDesc));
      Status = UsbIo->UsbGetDeviceDescriptor (UsbIo, &DevDesc);
      PrintStatus (L"UsbGetDeviceDescriptor", Status);
      if (!EFI_ERROR (Status)) {
        Print (L"  bLength=%u bDescriptorType=%u bcdUSB=0x%04x idVendor=0x%04x idProduct=0x%04x bDeviceClass=0x%02x\n",
               DevDesc.Length, DevDesc.DescriptorType, DevDesc.BcdUSB, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
      }
    }

    // 2) Get active Configuration Descriptor
    {
      // NOTE: some implementations allocate descriptor internally and return pointer; previous version used pointer.
      // Here we use stack-local struct version (UsbGetConfigDescriptor returns a pointer in original API; keep minimal)
      EFI_USB_CONFIG_DESCRIPTOR ConfigDesc;
      ZeroMem (&ConfigDesc, sizeof(ConfigDesc));
      Status = UsbIo->UsbGetConfigDescriptor (UsbIo, &ConfigDesc);
      PrintStatus (L"UsbGetConfigDescriptor", Status);
      if (!EFI_ERROR (Status)) {
        Print (L"  wTotalLength=%u bNumInterfaces=%u bConfigurationValue=%u\n",
               ConfigDesc.TotalLength, ConfigDesc.NumInterfaces, ConfigDesc.ConfigurationValue);
        UINTN DumpLen = (ConfigDesc.TotalLength > 64) ? 64 : ConfigDesc.TotalLength;
        Print(L" skipping DumpHex %u \n", DumpLen);
      }
    }

    // 3) Enumerate interfaces (try to get interface descriptor)
    {
      EFI_USB_INTERFACE_DESCRIPTOR IfDesc;
      ZeroMem (&IfDesc, sizeof(IfDesc));
      Status = UsbIo->UsbGetInterfaceDescriptor (UsbIo, &IfDesc);
      PrintStatus (L"UsbGetInterfaceDescriptor", Status);
      if (!EFI_ERROR (Status)) {
        Print (L"  Interface: bInterfaceNumber=%u bAlternateSetting=%u bInterfaceClass=0x%02x bInterfaceSubClass=0x%02x bInterfaceProtocol=0x%02x\n",
               IfDesc.InterfaceNumber, IfDesc.AlternateSetting, IfDesc.InterfaceClass, IfDesc.InterfaceSubClass, IfDesc.InterfaceProtocol);
      }
    }

    // 4) Enumerate endpoint descriptors (index until error)
    {
      UINT8 EpIndex = 0;
      EFI_STATUS EpStatus;
      EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
      while (TRUE) {
        ZeroMem (&EpDesc, sizeof(EpDesc));
        EpStatus = UsbIo->UsbGetEndpointDescriptor (UsbIo, EpIndex, &EpDesc);
        if (EFI_ERROR (EpStatus)) {
          if (EpIndex == 0) {
            Print (L"  UsbGetEndpointDescriptor returned %r (no endpoints?)\n", EpStatus);
          } else {
            Print (L"  End of endpoint list (index %u). Last status=%r\n", EpIndex, EpStatus);
          }
          break;
        }
        Print (L"  Endpoint %u: bEndpointAddress=0x%02x bmAttributes=0x%02x wMaxPacketSize=%u bInterval=%u\n",
               EpIndex, EpDesc.EndpointAddress, EpDesc.Attributes, EpDesc.MaxPacketSize, EpDesc.Interval);
        EpIndex++;
      }
    }


	// skipped this part - I have an errors here, and its freeze program
	// its returns names like Mass storage, or generic etc...

	/*
    // 5) Get supported language table (for strings)
    {
      UINT16 *LangTable = NULL;
      UINT16 TableSize = 0;
      Status = UsbIo->UsbGetSupportedLanguages (UsbIo, &LangTable, &TableSize);
      PrintStatus (L"UsbGetSupportedLanguages", Status);
      if (!EFI_ERROR (Status) && LangTable != NULL && TableSize > 0) {
        Print (L"  Supported languages count=%u first LangID=0x%04x\n", TableSize, LangTable[0]);
        // 6) Try to retrieve manufacturer & product strings (if string IDs are present)
        {
          CHAR16 *Str = NULL;
          Status = UsbIo->UsbGetStringDescriptor (UsbIo, LangTable[0], 1, &Str); // string ID 1 commonly manufacturer
          PrintStatus (L"  UsbGetStringDescriptor(ID=1)", Status);
          if (!EFI_ERROR (Status) && Str != NULL) {
            Print (L"    String[1] = %s\n", Str);
            FreePool (Str);
          }
		  
          Str = NULL;
          Status = UsbIo->UsbGetStringDescriptor (UsbIo, LangTable[0], 2, &Str); // string ID 2 commonly product
          PrintStatus (L"  UsbGetStringDescriptor(ID=2)", Status);
          if (!EFI_ERROR (Status) && Str != NULL) {
            Print (L"    String[2] = %s\n", Str);
            FreePool (Str);
          }
		  
        }
        FreePool (LangTable);
      }
    }
	*/

    // 7) Do a safe control transfer: GET_STATUS (2 bytes in)
    {
      EFI_USB_DEVICE_REQUEST DevReq;
      UINT8 StatusBuf[2];
      UINT32 UsbStatus = 0;
      ZeroMem (&DevReq, sizeof(DevReq));
      DevReq.RequestType = USB_ENDPOINT_DIR_IN | USB_REQ_TYPE_STANDARD | USB_TARGET_DEVICE;
      DevReq.Request     = USB_REQ_GET_STATUS;
      DevReq.Value       = 0;
      DevReq.Index       = 0;
      DevReq.Length      = 2;

      ZeroMem (StatusBuf, sizeof(StatusBuf));
      Status = UsbIo->UsbControlTransfer (
                        UsbIo,
                        &DevReq,
                        EfiUsbDataIn,
                        TIMEOUT_MS,
                        StatusBuf,
                        sizeof(StatusBuf),
                        &UsbStatus
                        );
      PrintStatus (L"UsbControlTransfer(GET_STATUS)", Status);
      if (!EFI_ERROR (Status)) {
        Print (L"  GET_STATUS returned UsbStatus=0x%x Data: ", UsbStatus);
        DumpHex (NULL, StatusBuf, sizeof(StatusBuf));
      }
    }

    // 8) Port reset (optional, safe only if you know the device): here we just attempt and print result
    {
      Status = UsbIo->UsbPortReset (UsbIo);
      PrintStatus (L"UsbPortReset (note: this may re-enumerate device)", Status);
    }

    Print (L"--- End of device %u checks ---\n", (UINT32)Index);
  }

  if (Handles != NULL) {
    FreePool (Handles);
  }
}

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  Print (L"UsbDiagApp: Starting USB diagnostic sequence\n");

  // Stage 1: scan PCI for USB controllers (B:D.F and class)
  ScanPciForUsbControllers ();

  // Stage 2: find HC protocols
  FindUsbHostControllerProtocols ();

  // Additional mapped checks:
  MapUsb2HcToPciBdf ();

  // EHCI register dump (read-only)
  ScanAllPciForEhciAndCheck ();

  // Stage 3: find and exercise EFI_USB_IO_PROTOCOL devices
  CheckUsbDevices ();

  // BlockIo checks and safe LBA0 read
  CheckUsbBlockIoAndReadLba0 ();

  Print (L"UsbDiagApp: Completed checks. You can re-run to capture new events.\n");

  // wait for key
  Print(L"Done. Press any key to exit...\n");
  EFI_INPUT_KEY Key;
  while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) != EFI_SUCCESS) {
    // spin
  }

  return EFI_SUCCESS;
}
