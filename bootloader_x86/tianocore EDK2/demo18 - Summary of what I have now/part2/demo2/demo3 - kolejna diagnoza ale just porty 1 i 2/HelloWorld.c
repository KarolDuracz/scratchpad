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

#include <Protocol/DevicePath.h>

#define DP_TYPE_END            0x7F
#define DP_SUBTYPE_END_ENTIRE  0xFF

#define DP_TYPE_HARDWARE       0x01
#define DP_TYPE_MESSAGING      0x03

#define DP_SUBTYPE_PCI         0x01   // PCI device path subtype (Hardware)
#define DP_SUBTYPE_USB         0x05   // USB device path subtype (Messaging)

#define MAX_PORTS_SAFE 32   // sanity cap for ports to avoid runaway loops

#define TIMEOUT_MS 5000
#define MAX_BARS 6
#define MAX_PORTS_TO_PRINT 16

//#define MAX_BARS 6
//#define MAX_PORTS_SAFE 32
#define FRINDEX_POLL_COUNT 5
#define FRINDEX_POLL_DELAY_MS 50

CHAR16 *
EFIAPI
MyDevicePathToStr (
  IN CONST EFI_DEVICE_PATH_PROTOCOL  *DevicePath
  )
{
  CONST EFI_DEVICE_PATH_PROTOCOL *Node;
  CHAR16 *Out;
  UINTN OutSizeChars = 1024;
  UINTN Pos = 0;

  if (DevicePath == NULL) {
    return NULL;
  }

  Out = AllocatePool (OutSizeChars * sizeof(CHAR16));
  if (Out == NULL) {
    return NULL;
  }
  Out[0] = L'\0';

  Node = DevicePath;
  while (Node != NULL) {
    UINT8 Type = Node->Type;
    UINT8 SubType = Node->SubType;
    UINT16 NodeLen = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));

    // End node?
    if (Type == DP_TYPE_END && SubType == DP_SUBTYPE_END_ENTIRE) {
      break;
    }
    // Guard: minimal node length is 4 (header)
    if (NodeLen < 4) {
      break;
    }

    // Ensure space (characters) for next append; if low, grow buffer
    if ((OutSizeChars - Pos) < 64) {
      UINTN NewSize = OutSizeChars * 2;
      CHAR16 *New = AllocatePool (NewSize * sizeof(CHAR16));
      if (New == NULL) break;
      CopyMem (New, Out, OutSizeChars * sizeof(CHAR16));
      FreePool (Out);
      Out = New;
      OutSizeChars = NewSize;
    }

    // Remaining size in bytes for UnicodeSPrint (it expects bytes)
    UINTN RemainingBytes = (OutSizeChars - Pos) * sizeof(CHAR16);

    if (Type == DP_TYPE_HARDWARE && SubType == DP_SUBTYPE_PCI && NodeLen >= 6) {
      // PCI node layout: Header (4) + UINT8 Device + UINT8 Function
      UINT8 *raw = (UINT8*)Node;
      UINT8 Device = raw[4];
      UINT8 Function = raw[5];
      // append "Pci(d,f)/"
      UnicodeSPrint (&Out[Pos], RemainingBytes, L"Pci(%u,%u)/", (UINTN)Device, (UINTN)Function);
      Pos += StrLen (&Out[Pos]);
    } else if (Type == DP_TYPE_MESSAGING && SubType == DP_SUBTYPE_USB && NodeLen >= 6) {
      // USB node layout: Header (4) + UINT8 ParentPortNumber + UINT8 InterfaceNumber
      UINT8 *raw = (UINT8*)Node;
      UINT8 ParentPort = raw[4];
      UINT8 Interface = raw[5];
      // append "USB(port,iface)/"
      UnicodeSPrint (&Out[Pos], RemainingBytes, L"USB(%u,%u)/", (UINTN)ParentPort, (UINTN)Interface);
      Pos += StrLen (&Out[Pos]);
    } else {
      // Generic node print: "Txx.Syy/"
      UnicodeSPrint (&Out[Pos], RemainingBytes, L"T%02x.S%02x/", (UINTN)Type, (UINTN)SubType);
      Pos += StrLen (&Out[Pos]);
    }

    // advance to next node
    Node = (CONST EFI_DEVICE_PATH_PROTOCOL *) ((CONST UINT8 *) Node + NodeLen);
  }

  // Ensure not empty
  if (Pos == 0) {
    UnicodeSPrint (Out, OutSizeChars * sizeof(CHAR16), L"(empty device path)");
  }

  return Out;
}

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

// few anothers helpers
STATIC
CONST CHAR16 *
KnownProtocolName (
  IN EFI_GUID *ProtocolGuid
  )
{
  // Compare common GUIDs you care about and return readable names.
  // Add more mappings if needed.
  if (CompareGuid (ProtocolGuid, &gEfiUsbIoProtocolGuid))            return L"EFI_USB_IO_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiBlockIoProtocolGuid))          return L"EFI_BLOCK_IO_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiSimpleFileSystemProtocolGuid)) return L"EFI_SIMPLE_FILE_SYSTEM_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiSimplePointerProtocolGuid))    return L"EFI_SIMPLE_POINTER_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiSimpleTextInProtocolGuid))     return L"EFI_SIMPLE_TEXT_IN_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiUsb2HcProtocolGuid))           return L"EFI_USB2_HC_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiUsbHcProtocolGuid))            return L"EFI_USB_HC_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiPciIoProtocolGuid))            return L"EFI_PCI_IO_PROTOCOL";
  // fallback
  return NULL;
}

STATIC
VOID
ListHandlesAndProtocols (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN i;

  Print (L"--- Dump handles (DevicePath) and installed protocols ---\n");

  // Locate handles that have a DevicePath (most device handles do)
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiDevicePathProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR (Status) || HandleCount == 0) {
    Print (L"  LocateHandleBuffer(DevicePath) failed or none: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HandleCount; ++i) {
    EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
    CHAR16 *DpStr = NULL;
    EFI_GUID **ProtocolArray = NULL;   // correct type: array of CONST EFI_GUID*
    UINTN ProtocolCount = 0;
    UINTN j;

	//EFI_DEVICE_PATH_PROTOCOL *DevPath;
	//CHAR16 *DpStr;
	Status = gBS->HandleProtocol (Handles[i], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
	if (!EFI_ERROR (Status) && DevPath != NULL) {
	  DpStr = MyDevicePathToStr (DevPath);
	  if (DpStr != NULL) {
		Print(L"DevicePath=%s\n", DpStr);
		FreePool(DpStr);
	  }
	}


	    //
    // SAFE: do not read DevicePath nodes here (suppressed). Just list protocols
    // without dereferencing GUID pointers (which may be invalid on some firmwares).
    //
    Print (L"Handle %u: DevicePath=(suppressed for stability)\n", (UINT32)i);

	//CONST EFI_GUID **ProtocolArray = NULL;
	//UINTN ProtocolCount = 0;
	//UINTN j;

	Status = gBS->ProtocolsPerHandle(Handles[i], (EFI_GUID ***)&ProtocolArray, &ProtocolCount);
	if (!EFI_ERROR(Status) && ProtocolArray != NULL) {
	  if (ProtocolCount > 1024) ProtocolCount = 1024; // safety cap
	  for (j = 0; j < ProtocolCount; ++j) {
		CONST EFI_GUID *g = ProtocolArray[j];
		if (g == NULL) {
		  Print(L"  [%u] NULL\n", (UINT32)j);
		  continue;
		}
		if (g == &gEfiUsbIoProtocolGuid) {
		  Print(L"  [%u] EFI_USB_IO_PROTOCOL\n", (UINT32)j);
		} else {
		  Print(L"  [%u] Protocol ptr=%p\n", (UINT32)j, (VOID*)g);
		}
	  }
	  FreePool((VOID*)ProtocolArray);
	}



	/*
    // ProtocolsPerHandle returns CONST EFI_GUID **ProtocolArray (allocated by the firmware)
    Status = gBS->ProtocolsPerHandle (Handles[i], (EFI_GUID ***)&ProtocolArray, &ProtocolCount);
    if (EFI_ERROR (Status)) {
      Print (L"  ProtocolsPerHandle failed: 0x%lx\n", Status);
      continue;
    }

    // Sanity cap for protocol count to avoid runaway values from buggy firmware
    if (ProtocolCount > 1024) {
      Print (L"  ProtocolCount suspicious (%u). Capping to 1024 for safety.\n", (UINT32)ProtocolCount);
      ProtocolCount = 1024;
    }

    for (j = 0; j < ProtocolCount; ++j) {
      EFI_GUID *GuidPtr = ProtocolArray[j];

      if (GuidPtr == NULL) {
        Print (L"    [%03u] <NULL GUID pointer>\n", (UINT32)j);
        continue;
      }

      //
      // Safe name detection: compare pointer values to known GUID variables.
      // This does NOT dereference the GUID at GuidPtr.
      //
      if (GuidPtr == &gEfiUsbIoProtocolGuid) {
        Print (L"    [%03u] EFI_USB_IO_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiBlockIoProtocolGuid) {
        Print (L"    [%03u] EFI_BLOCK_IO_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiSimpleFileSystemProtocolGuid) {
        Print (L"    [%03u] EFI_SIMPLE_FILE_SYSTEM_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiSimplePointerProtocolGuid) {
        Print (L"    [%03u] EFI_SIMPLE_POINTER_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiSimpleTextInProtocolGuid) {
        Print (L"    [%03u] EFI_SIMPLE_TEXT_IN_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiUsb2HcProtocolGuid) {
        Print (L"    [%03u] EFI_USB2_HC_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiUsbHcProtocolGuid) {
        Print (L"    [%03u] EFI_USB_HC_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiPciIoProtocolGuid) {
        Print (L"    [%03u] EFI_PCI_IO_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else {
        // Unknown protocol: print the pointer value instead of dereferencing contents
        Print (L"    [%03u] Protocol ptr=%016lx (unknown)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      }
    }
	*/

    // Free the allocated ProtocolArray buffer (allocated by ProtocolsPerHandle)
    if (ProtocolArray != NULL) {
      FreePool ((VOID*)ProtocolArray);
      ProtocolArray = NULL;
    }



	/*
	    //
    // DO NOT call HandleProtocol(..., &gEfiDevicePathProtocolGuid, ...)
    // and DO NOT call DevicePathToStr() or MyDevicePathToStr() here --
    // those were triggering exceptions on your platform. Instead
    // simply print a placeholder and continue listing the protocols.
    //
    Print (L"Handle %u: DevicePath=(suppressed for stability)\n", (UINT32)i);

    // Get list of protocols installed on this handle
    // ProtocolsPerHandle will allocate ProtocolArray for us.
    Status = gBS->ProtocolsPerHandle (Handles[i], (EFI_GUID ***)&ProtocolArray, &ProtocolCount);
    if (EFI_ERROR (Status)) {
      Print (L"  ProtocolsPerHandle failed: 0x%lx\n", Status);
      // Do not attempt to free any DevicePath string (we never allocated it).
      continue;
    }

    // Defensive printing: do not read beyond ProtocolCount; check pointers.
    for (j = 0; j < ProtocolCount; ++j) {
      EFI_GUID *Guid = ProtocolArray[j];
      if (Guid == NULL) {
        Print (L"    [%03u] <NULL GUID pointer>\n", (UINT32)j);
        continue;
      }

      CONST CHAR16 *Name = KnownProtocolName (Guid);
      if (Name != NULL) {
        Print (L"    [%03u] %s\n", (UINT32)j, Name);
      } else {
        // Print raw GUID safely
        Print (L"    [%03u] GUID=%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
               (UINT32)j,
               (UINT32)Guid->Data1,
               (UINT32)Guid->Data2,
               (UINT32)Guid->Data3,
               (UINT32)Guid->Data4[0], (UINT32)Guid->Data4[1],
               (UINT32)Guid->Data4[2], (UINT32)Guid->Data4[3],
               (UINT32)Guid->Data4[4], (UINT32)Guid->Data4[5],
               (UINT32)Guid->Data4[6], (UINT32)Guid->Data4[7]);
      }
    }

    // Free the protocol array buffer allocated by ProtocolsPerHandle
    if (ProtocolArray != NULL) {
      FreePool ((VOID*)ProtocolArray);
      ProtocolArray = NULL;
    }
	*/


	/*
    Status = gBS->HandleProtocol (Handles[i], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
    if (!EFI_ERROR (Status) && DevPath != NULL) {
      DpStr = DevicePathToStr (DevPath);     // returns CHAR16*
      if (DpStr == NULL) {
        Print (L"Handle %u: DevicePathToStr returned NULL\n", (UINT32)i);
      }
    }
	*/

    //Print (L"Handle %u: DevicePath=%s\n", (UINT32)i, DpStr ? DpStr : L"(no device path)");
	
	/*
    // Get list of protocols installed on this handle
    Status = gBS->ProtocolsPerHandle (Handles[i], &ProtocolArray, &ProtocolCount);
    if (EFI_ERROR (Status)) {
      Print (L"  ProtocolsPerHandle failed: 0x%lx\n", Status);
      if (DpStr) FreePool (DpStr);
      continue;
    }

    for (j = 0; j < ProtocolCount; ++j) {
      EFI_GUID *Guid = ProtocolArray[j];
      CONST CHAR16 *Name = KnownProtocolName (Guid);
      if (Name != NULL) {
        Print (L"    [%03u] %s\n", (UINT32)j, Name);
      } else {
        // Print raw GUID
        Print (L"    [%03u] GUID=%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
               (UINT32)j,
               Guid->Data1, Guid->Data2, Guid->Data3,
               Guid->Data4[0], Guid->Data4[1],
               Guid->Data4[2], Guid->Data4[3], Guid->Data4[4], Guid->Data4[5], Guid->Data4[6], Guid->Data4[7]);
      }
    }

    // Free the protocol array buffer allocated by ProtocolsPerHandle
    if (ProtocolArray) {
      FreePool ((VOID*)ProtocolArray);
      ProtocolArray = NULL;
    }
	*/

    if (DpStr) {
      FreePool (DpStr);
      DpStr = NULL;
    }
  }

  FreePool (Handles);
}

/*
STATIC
BOOLEAN
ParseUsbPortFromDevicePathStr (
  IN CHAR16 *DpStr,
  OUT UINTN *PortNumber  // first port field
  )
{
  // Simple search for "USB(" and parse "USB(n" where n is decimal.
  CHAR16 *p = DpStr;
  while (p != NULL && *p != L'\0') {
    p = StrStr (p, L"USB(");
    if (p == NULL) {
      break;
    }
    p += 4; // point at first char after "USB("
    if ((*p) >= L'0' && (*p) <= L'9') {
      UINTN val = 0;
      while ((*p) >= L'0' && (*p) <= L'9') {
        val = val * 10 + (UINTN)(*p - L'0');
        p++;
      }
      *PortNumber = val;
      return TRUE;
    }
  }
  return FALSE;
}
*/

STATIC
VOID
PrintPortScSummary (
  IN UINT32 PortSc,
  IN UINT32 PortIndex
  )
{
  UINTN CCS = (PortSc >> 0) & 1;
  UINTN CSC = (PortSc >> 1) & 1;
  UINTN PE  = (PortSc >> 2) & 1;
  UINTN PE_CHANGE = (PortSc >> 3) & 1;
  UINTN PortOwner = (PortSc >> 13) & 1; // spec-dependent bit pos; used as indicative
  // simplified speed decode (specs vary by controller)
  UINTN PortSpeed = (PortSc >> 26) & 3;

  Print (L"      PORT%u: PORTSC raw=0x%08x  (CCS=%u CSC=%u PE=%u PEchg=%u PO=%u Speed=%u)\n",
         (UINT32)PortIndex, PortSc, (UINT32)CCS, (UINT32)CSC, (UINT32)PE, (UINT32)PE_CHANGE, (UINT32)PortOwner, (UINT32)PortSpeed);
}

STATIC
VOID
ExamineUsbPortEnumerationAndCompanionOwnership (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN PciCount = 0;
  UINTN i;

  Print (L"--- Examining USB controllers: ports vs published EFI_USB_IO_PROTOCOL handles ---\n");

  // Count all UsbIo handles in system (global)
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbHandleCount = 0;
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbHandleCount, &UsbHandles);
  if (EFI_ERROR (Status)) {
    UsbHandleCount = 0;
  }
  Print (L"Total EFI_USB_IO_PROTOCOL handles in system: %u\n", (UINT32)UsbHandleCount);

  // Locate all PCI handles
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(PciIo) failed: 0x%lx\n", Status);
    if (UsbHandles) FreePool (UsbHandles);
    return;
  }

  for (i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo;
    UINT32 Data32 = 0;
    UINTN Bus, Device, Function, Segment;

    Status = gBS->HandleProtocol (PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR (Status) || PciIo == NULL) {
      continue;
    }

    PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);

    // Read class/subclass/progif from config dword at offset 0x08
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &Data32);
    if (EFI_ERROR (Status)) {
      continue;
    }
    UINT8 ClassCode = (UINT8)((Data32 >> 24) & 0xFF);
    UINT8 SubClass  = (UINT8)((Data32 >> 16) & 0xFF);
    UINT8 ProgIf    = (UINT8)((Data32 >> 8) & 0xFF);

    if (!(ClassCode == 0x0C && SubClass == 0x03)) {
      // Not a USB host controller
      continue;
    }

    Print (L"USB controller at PCI %02x:%02x.%x  ProgIf=0x%02x\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function, ProgIf);

    // Probe BARs to find capability base (CAPLENGTH at offset 0x00 in MMIO)
    BOOLEAN FoundMmio = FALSE;
    for (UINT8 BarIndex = 0; BarIndex < MAX_BARS && !FoundMmio; ++BarIndex) {
      UINT8 CapLength = 0xFF;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLength);
      if (EFI_ERROR (Status)) {
        continue;
      }
      if (CapLength == 0xFF || CapLength == 0x00) {
        continue;
      }

      // Found likely capability region
      FoundMmio = TRUE;
      UINTN OpBase = (UINTN)CapLength;

      // Read USBCMD, USBSTS
      UINT32 UsbCmd = 0, UsbSts = 0, HcSParams = 0;
      PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
      PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x04, 1, &UsbSts); // sometimes HCSPARAMS is at cap offset 0x04
      // For safety also read HCSPARAMS from capability offset 0x04
      PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);

      Print (L"  BAR%u CAPLENGTH=0x%02x  USBCMD=0x%08x USBSTS=0x%08x HCSPARAMS(raw)=0x%08x\n",
             (UINT32)BarIndex, CapLength, UsbCmd, UsbSts, HcSParams);

      // Determine number of ports (specs vary; we use lower 4 bits as approximate)
      UINT32 NPorts = (HcSParams & 0x0000000F);
      if (NPorts == 0) {
        // Some controllers provide bits differently; fall back to safe max
        NPorts = MAX_PORTS_SAFE;
      }
      if (NPorts > MAX_PORTS_SAFE) {
        NPorts = MAX_PORTS_SAFE;
      }

      // count connected ports for this controller
      UINT32 ConnectedCount = 0;

      for (UINT32 PortIndex = 1; PortIndex <= NPorts; ++PortIndex) {
        // PORTSC offset in operational regs: 0x44 + 4*(port-1)
        UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
        UINT32 PortSc = 0;
        EFI_STATUS r = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
        if (EFI_ERROR (r)) {
          // stop if read fails (no more ports)
          break;
        }
        // Decode relevant bits
        UINTN CCS = (PortSc >> 0) & 1;
        UINTN PO  = (PortSc >> 13) & 1; // example PO bit location (often bit 13)
        UINTN PE  = (PortSc >> 2) & 1;
        if (CCS) {
          ConnectedCount++;
        }
        // Print port summary
        PrintPortScSummary (PortSc, PortIndex);

        if (CCS && PO) {
          Print (L"        -> Port%u connected but PO (Port Owner) set: companion controller likely owns this port.\n", (UINT32)PortIndex);
        }

        if (CCS && !PE) {
          Print (L"        -> Port%u connected but not enabled (PE=0). Enumeration may not have completed.\n", (UINT32)PortIndex);
        }
      } // per-port loop

      Print (L"  Summary for controller %02x:%02x.%x: ConnectedPorts=%u\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function, ConnectedCount);

      // Compare to global count of UsbIo handles (coarse check)
      if (ConnectedCount > 0 && (UINT32)UsbHandleCount == 0) {
        Print (L"  Warning: %u physical port(s) show connected devices but there are 0 EFI_USB_IO_PROTOCOL handles in the system.\n", ConnectedCount);
        Print (L"           Likely causes: missing USB bus driver, companion controller owning ports (PO=1), or controller halted.\n");
      } else if ((UINT32)UsbHandleCount < ConnectedCount) {
        Print (L"  Notice: %u connected ports vs %u UsbIo device handles overall. Some devices may be un-enumerated.\n",
               ConnectedCount, (UINT32)UsbHandleCount);
      } else {
        Print (L"  Device-handle count (%u) >= connected ports (%u) -- enumeration seems to have succeeded globally.\n",
               (UINT32)UsbHandleCount, ConnectedCount);
      }

      // HC state diagnostics
      // USBCMD bit 0 = Run/Stop (1=run), USBSTS bit 12 = HCHalted (1=halted) (bit positions may vary by spec)
      UINTN RunStop = UsbCmd & 1;
      UINTN HCHalted = (UsbSts >> 12) & 1;
      if (!RunStop) {
        Print (L"  Host Controller appears stopped (USBCMD.Run/Stop = 0). The HC driver may not have started the schedule.\n");
      }
      if (HCHalted) {
        Print (L"  Host Controller reports halted (USBSTS.HCHalted = 1). HC reset/driver issues may exist.\n");
      }

      // done with this BAR / controller
    } // per BAR

    if (!FoundMmio) {
      Print (L"  No accessible MMIO BAR found for this controller (skipped).\n");
    }
  } // per PCI handle

  if (PciHandles) FreePool (PciHandles);
  if (UsbHandles) FreePool (UsbHandles);

  Print (L"--- End of port enumeration vs handles check ---\n");
}

//
// Defensive diagnostic: when PORT shows CCS=1 (connected), PE=0 (not enabled),
// and PO=0 (not owned by companion), call a deeper diagnostic function to inspect
// controller internal registers and schedule state.
//
// Usage: call CheckUnenumeratedPortsAndDiagnose() from UefiMain.
//
// Requires:
//   <Uefi.h>, <Library/UefiLib.h>, <Library/UefiBootServicesTableLib.h>,
//   <Protocol/PciIo.h>
//



STATIC
VOID
DiagnosePortOnController (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               OpBase,
  IN UINT32              PortIndex
  )
{
  if (PciIo == NULL) {
    Print (L"  DiagnosePortOnController: PciIo is NULL\n");
    return;
  }

  EFI_STATUS Status;
  UINT32 UsbCmd = 0;
  UINT32 UsbSts = 0;
  UINT32 FrIndex = 0;
  UINT32 AsyncList = 0;
  UINT32 PeriodicBase = 0;
  UINT32 HcSParams = 0;
  UINT32 PortSc = 0;

  Print (L"  --- Diagnose: controller BAR%u OpBase=0x%03x port=%u ---\n", (UINT32)BarIndex, (UINT32)OpBase, (UINT32)PortIndex);

  // Read USBCMD (operational offset 0x00)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
  if (EFI_ERROR (Status)) {
    Print (L"    Read USBCMD failed: 0x%lx\n", Status);
    return;
  }

  // Read USBSTS (operational offset 0x04)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x04, 1, &UsbSts);
  if (EFI_ERROR (Status)) {
    Print (L"    Read USBSTS failed: 0x%lx\n", Status);
    return;
  }

  // Read FRINDEX (op offset 0x0C)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x0C, 1, &FrIndex);
  if (EFI_ERROR (Status)) {
    Print (L"    Read FRINDEX failed: 0x%lx\n", Status);
    // continue - FRINDEX is useful but not mandatory
    FrIndex = 0xFFFFFFFF;
  }

  // Read ASYNCLISTADDR (op offset 0x18)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x18, 1, &AsyncList);
  if (EFI_ERROR (Status)) {
    Print (L"    Read ASYNCLISTADDR failed: 0x%lx\n", Status);
    AsyncList = 0;
  }

  // Read PERIODICLISTBASE (op offset 0x10)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x10, 1, &PeriodicBase);
  if (EFI_ERROR (Status)) {
    Print (L"    Read PERIODICLISTBASE failed: 0x%lx\n", Status);
    PeriodicBase = 0;
  }

  // Read HCSPARAMS (capability offset 0x04)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);
  if (EFI_ERROR (Status)) {
    Print (L"    Read HCSPARAMS failed: 0x%lx\n", Status);
    HcSParams = 0;
  }

  // Read PORTSC for this port
  UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
  if (EFI_ERROR (Status)) {
    Print (L"    Read PORT%u failed: 0x%lx\n", PortIndex, Status);
    return;
  }

  // Print basic register state
  Print (L"    USBCMD=0x%08x USBSTS=0x%08x FRINDEX=0x%08x\n", UsbCmd, UsbSts, FrIndex);
  Print (L"    ASYNCLISTADDR=0x%08x PERIODICLISTBASE=0x%08x HCSPARAMS=0x%08x\n", AsyncList, PeriodicBase, HcSParams);
  PrintPortScSummary (PortSc, PortIndex);

  // Decode suspicious cases
  UINTN RunStop = UsbCmd & 1;              // USBCMD bit 0
  UINTN HCHalted = (UsbSts >> 12) & 1;     // USBSTS.HCHalted (bit 12)
  UINTN AsyncEnabled = (UsbCmd >> 5) & 1;  // USBCMD Async Schedule Enable (bit 5)
  UINTN PeriodicEnabled = (UsbCmd >> 4) & 1; // USBCMD Periodic Schedule Enable (bit 4)

  if (!RunStop) {
    Print (L"    NOTE: Host controller Run/Stop == 0 (controller stopped). Driver may not have started schedules.\n");
  }
  if (HCHalted) {
    Print (L"    WARNING: HCHalted == 1 (controller halted). Needs driver reset or investigation.\n");
  }
  if (AsyncList == 0) {
    Print (L"    NOTE: ASYNCLISTADDR == 0 (no async list programmed). That prevents transfers for devices using async schedules.\n");
  } else {
    Print (L"    ASYNCLISTADDR points to 0x%08x\n", AsyncList);
  }
  if (!AsyncEnabled) {
    Print (L"    NOTE: Async schedule not enabled (USBCMD.AsyncEnable=0).\n");
  }
  if (!PeriodicEnabled) {
    Print (L"    NOTE: Periodic schedule not enabled (USBCMD.PeriodicEnable=0).\n");
  }

  // FRINDEX sanity: poll a few times to see if it advances (schedules running)
  if (FrIndex != 0xFFFFFFFF) {
    UINT32 prev = FrIndex;
    UINT32 cur = prev;
    UINTN advanced = 0;
    for (UINTN k = 0; k < FRINDEX_POLL_COUNT; ++k) {
      // delay
      gBS->Stall (FRINDEX_POLL_DELAY_MS * 1000); // Stall expects microseconds
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x0C, 1, &cur);
      if (EFI_ERROR (Status)) {
        Print (L"    Read FRINDEX failed during poll: 0x%lx\n", Status);
        cur = prev;
        break;
      }
      if (cur != prev) {
        advanced = 1;
        break;
      }
    }
    if (advanced) {
      Print (L"    FRINDEX advanced -> schedule appears to be running.\n");
    } else {
      Print (L"    FRINDEX did not advance in %u polls -> schedule may be stalled.\n", FRINDEX_POLL_COUNT);
    }
  } else {
    Print (L"    FRINDEX not available; cannot test schedule activity.\n");
  }

  // Check PORTSC change bits for hints (CSC, PED, etc.)
  UINTN CSC = (PortSc >> 1) & 1;
  UINTN PE   = (PortSc >> 2) & 1;
  UINTN PO   = (PortSc >> 13) & 1;
  UINTN CCS  = (PortSc >> 0) & 1;

  if (CCS && !PE && !PO) {
    Print (L"    Diagnosis: Connected but not enabled and EHCI owns the port (PO=0) -> likely HC driver or scheduling problem.\n");
    Print (L"    Suggested next steps: check HC driver presence, ensure Async list programmed and Async schedule enabled; enable debug logs in bus/HC driver.\n");
  }
  
  if (!CSC) {
	  Print(L" CSC is here but is not read anywhere \n ");
  }

  // Read a second snapshot of PORTSC to compare (non-destructive)
  UINT32 PortSc2 = 0;
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc2);
  if (!EFI_ERROR (Status)) {
    if (PortSc2 != PortSc) {
      Print (L"    PORTSC changed during diagnosis: old=0x%08x new=0x%08x\n", PortSc, PortSc2);
    } else {
      Print (L"    PORTSC unchanged during diagnosis.\n");
    }
  } else {
    Print (L"    Failed to re-read PORTSC: 0x%lx\n", Status);
  }

  Print (L"  --- End diagnose for port %u ---\n", PortIndex);
}

STATIC
VOID
CheckUnenumeratedPortsAndDiagnose (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN HandleCount = 0;

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &PciHandles);
  if (EFI_ERROR (Status)) {
    Print (L"CheckUnenumeratedPortsAndDiagnose: LocateHandleBuffer(PciIo) failed: 0x%lx\n", Status);
    return;
  }

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol (PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR (Status) || PciIo == NULL) {
      continue;
    }

    // Read class/subclass/progif to ensure it's a USB host controller
    UINT32 ClassDword = 0;
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassDword);
    if (EFI_ERROR (Status)) {
      continue;
    }
    UINT8 ClassCode = (UINT8)((ClassDword >> 24) & 0xFF);
    UINT8 SubClass  = (UINT8)((ClassDword >> 16) & 0xFF);
    if (!(ClassCode == 0x0C && SubClass == 0x03)) {
      continue; // skip non-USB controllers
    }

    // Probe BARs to locate capability/op region
    for (UINT8 BarIndex = 0; BarIndex < MAX_BARS; ++BarIndex) {
      UINT8 CapLen = 0xFF;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLen);
      if (EFI_ERROR (Status)) {
        continue;
      }
      if (CapLen == 0xFF || CapLen == 0x00) {
        continue;
      }

      UINTN OpBase = (UINTN)CapLen;
      // Read HCSPARAMS from capability offset 0x04
      UINT32 HcSParams = 0;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);
      if (EFI_ERROR (Status)) {
        // fallback: assume some ports but continue defensively
        HcSParams = 0;
      }

      UINT32 NPorts = (HcSParams & 0x0000000F);
      if (NPorts == 0) {
        NPorts = MAX_PORTS_SAFE;
      }
      if (NPorts > MAX_PORTS_SAFE) {
        NPorts = MAX_PORTS_SAFE;
      }

      for (UINT32 PortIndex = 1; PortIndex <= NPorts; ++PortIndex) {
        UINT32 PortSc = 0;
        UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
        Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
        if (EFI_ERROR (Status)) {
          break; // stop per-port loop (no more ports or read error)
        }

        UINTN CCS = (PortSc >> 0) & 1;
        UINTN PE  = (PortSc >> 2) & 1;
        UINTN PO  = (PortSc >> 13) & 1; // approximate bit pos; ok for diagnostics

        if (CCS && !PE && !PO) {
          // Found candidate un-enumerated port (connected, not enabled, EHCI owns)
          Print (L"Controller PCI BAR%u: Found port %u with CCS=1 PE=0 PO=0 -> diagnosing deeper\n", (UINT32)BarIndex, (UINT32)PortIndex);
          DiagnosePortOnController (PciIo, (UINT8)BarIndex, OpBase, PortIndex); // cast on UINT8 is my change. This was not produce by GPT-5 !!!! 
          // continue checking other ports (don't break)
        }
      } // per port loop
    } // per BAR loop
  } // per PCI handle

  if (PciHandles) {
    FreePool (PciHandles);
  }
}


// UEFI MAIN /////////////////////
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
  
		// anothers
		//ListHandlesAndProtocols ();
		ExamineUsbPortEnumerationAndCompanionOwnership();
		
		CheckUnenumeratedPortsAndDiagnose();
		

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
