/*
  UsbHostInfo.c

  Demo UEFI application to enumerate PCI USB Host Controllers and print
  detailed information (PCI IDs, BARs, interrupt, location) and host USB info
  (capabilities, root-hub port status). Also maps EFI_USB_IO handles to the
  PCI host controllers (device-path prefix match) and prints basic VID/PID.

  Build in EDK2 (MdeModulePkg/Application). Paste as a single .c file.
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/PrintLib.h>

#include <Protocol/PciIo.h>
#include <Protocol/DevicePath.h>
#include <Protocol/DevicePathToText.h>
#include <Protocol/Usb2HostController.h>
#include <Protocol/UsbIo.h>
#include <IndustryStandard/Usb.h>

#ifndef USB_PORT_STAT_CONNECTION
#define USB_PORT_STAT_CONNECTION 0x00000001
#endif

#ifndef END_DEVICE_PATH_TYPE
#define END_DEVICE_PATH_TYPE 0x7F
#endif
#ifndef END_ENTIRE_DEVICE_PATH_SUBTYPE
#define END_ENTIRE_DEVICE_PATH_SUBTYPE 0xFF
#endif

STATIC
UINTN
GetDevicePathSizeBytes (
  IN EFI_DEVICE_PATH_PROTOCOL  *DevicePath
  )
{
  EFI_DEVICE_PATH_PROTOCOL *Node;
  UINTN Size = 0;

  if (DevicePath == NULL) {
    return 0;
  }

  Node = DevicePath;
  while (TRUE) {
    UINT8  *Raw = (UINT8 *)Node;
    UINT16 Len = *(UINT16 *)(Raw + 2);
    if (Len < sizeof(EFI_DEVICE_PATH_PROTOCOL)) {
      return Size;
    }
    Size += Len;
    if (Node->Type == END_DEVICE_PATH_TYPE && Node->SubType == END_ENTIRE_DEVICE_PATH_SUBTYPE) {
      break;
    }
    Node = (EFI_DEVICE_PATH_PROTOCOL *)((UINT8 *)Node + Len);
  }

  return Size;
}

STATIC
BOOLEAN
DevicePathIsPrefixBinary (
  IN EFI_DEVICE_PATH_PROTOCOL *Prefix,
  IN UINTN                    PrefixSize,
  IN EFI_DEVICE_PATH_PROTOCOL *Path,
  IN UINTN                    PathSize
  )
{
  if (Prefix == NULL || Path == NULL) {
    return FALSE;
  }
  if (PrefixSize == 0 || PathSize < PrefixSize) {
    return FALSE;
  }
  return (CompareMem((VOID *)Prefix, (VOID *)Path, PrefixSize) == 0);
}

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN PciCount = 0;

  // Find all PCI handles
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Print(L"No PCI handles found (or error: %r)\n", Status);
    return EFI_SUCCESS;
  }

  // Prepare DevicePath->Text
  EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpToText = NULL;
  gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID **)&DpToText);

  Print(L"Scanning %u PCI devices for USB host controllers...\n\n", (UINT32)PciCount);

  // We'll collect PCI host controllers info into arrays to later map USB_IO handles
  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) {
      continue;
    }

    // Read class/prog-if at offset 0x08 (DWORD)
    UINT32 ClassReg = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassReg);
    if (EFI_ERROR(Status)) {
      continue;
    }

    UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
    UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
    UINT8 ProgIf    = (ClassReg >> 8)  & 0xFF;

    // Interested only in USB host controllers: BaseClass=0x0C (Serial Bus), SubClass=0x03 (USB)
    if (!(BaseClass == 0x0C && SubClass == 0x03)) {
      continue;
    }

    Print(L"PCI USB Host Controller found (handle=0x%p): Class=0x%02x Sub=0x%02x ProgIf=0x%02x\n",
          PciHandles[i], (UINT32)BaseClass, (UINT32)SubClass, (UINT32)ProgIf);

    // Print PCI location if available
    UINTN SegmentNumber = 0;
    UINTN BusNumber = 0;
    UINTN DeviceNumber = 0;
    UINTN FunctionNumber = 0;
    if (PciIo->GetLocation != NULL) {
      Status = PciIo->GetLocation(PciIo, &SegmentNumber, &BusNumber, &DeviceNumber, &FunctionNumber);
      if (!EFI_ERROR(Status)) {
        Print(L"  PCI Location: Seg=%u Bus=%u Dev=%u Func=%u\n",
              SegmentNumber, BusNumber, DeviceNumber, FunctionNumber);
      }
    }

    // Read VendorID/DeviceID (offset 0x00)
    UINT32 VendorDevice = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x00, 1, &VendorDevice);
    if (!EFI_ERROR(Status)) {
      UINT16 VendorId = (UINT16)(VendorDevice & 0xFFFF);
      UINT16 DeviceId = (UINT16)((VendorDevice >> 16) & 0xFFFF);
      Print(L"  PCI VendorId=0x%04x DeviceId=0x%04x\n", VendorId, DeviceId);
    }

    // Read header type + BARs (0x10..0x24)
    for (UINTN barOff = 0x10; barOff <= 0x24; barOff += 4) {
      UINT32 BarVal = 0;
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)barOff, 1, &BarVal);
      if (!EFI_ERROR(Status)) {
        Print(L"  BAR 0x%02x = 0x%08x\n", (UINT32)barOff, BarVal);
      }
    }

    // Read interrupt line/pin at 0x3C (byte) and 0x3D (pin)
    UINT8 IntLine = 0;
    UINT8 IntPin = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint8, 0x3C, 1, &IntLine);
    if (!EFI_ERROR(Status)) {
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint8, 0x3D, 1, &IntPin);
      Print(L"  Interrupt Line=0x%02x  Pin=0x%02x\n", IntLine, IntPin);
    }

    // Show textual device path if possible
    EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
    if (!EFI_ERROR(Status) && PciDp != NULL && DpToText != NULL) {
      CHAR16 *Txt = DpToText->ConvertDevicePathToText(PciDp, FALSE, FALSE);
      if (Txt) {
        Print(L"  PCI DevicePath: %s\n", Txt);
        FreePool(Txt);
      }
    }

    // Try to connect controller so the stack enumerates root hub/children
    Status = gBS->ConnectController(PciHandles[i], NULL, NULL, TRUE);
    if (EFI_ERROR(Status)) {
      Print(L"  ConnectController returned: %r\n", Status);
    } else {
      Print(L"  ConnectController: OK\n");
    }

    // Try to get USB2_HC protocol on this handle
    EFI_USB2_HC_PROTOCOL *Usb2Hc = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Usb2Hc);
    if (EFI_ERROR(Status) || Usb2Hc == NULL) {
      Print(L"  EFI_USB2_HC_PROTOCOL not present on this handle (status=%r)\n", Status);
    } else {
      UINT8 MaxSpeed = 0;
      UINT8 NumPorts = 0;
      UINT8 Is64 = 0;
      Status = Usb2Hc->GetCapability(Usb2Hc, &MaxSpeed, &NumPorts, &Is64);
      if (EFI_ERROR(Status)) {
        Print(L"  Usb2Hc->GetCapability failed: %r\n", Status);
      } else {
        Print(L"  Host Capabilities: maxSpeed=%u numPorts=%u 64bit=%u\n",
              (UINT32)MaxSpeed, (UINT32)NumPorts, (UINT32)Is64);

        // Query each root hub port status (some implementations return PortStatus only)
        for (UINT8 port = 1; port <= NumPorts; ++port) {
          EFI_USB_PORT_STATUS PortStatus;
          Status = Usb2Hc->GetRootHubPortStatus(Usb2Hc, port, &PortStatus);
          if (EFI_ERROR(Status)) {
            Print(L"    Port %u: GetRootHubPortStatus failed: %r\n", (UINT32)port, Status);
          } else {
            BOOLEAN connected = (PortStatus.PortStatus & USB_PORT_STAT_CONNECTION) != 0;
            Print(L"    Port %u: Connected=%u  PortStatus=0x%08x\n",
                  (UINT32)port, connected ? 1 : 0, (UINT32)PortStatus.PortStatus);
          }
        }
      }
    }

    Print(L"\n");
  } // end for each pci handle

  //
  // After attempting to connect controllers, find all USB_IO handles and map them to PCI hosts.
  //
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
  if (EFI_ERROR(Status) || UsbCount == 0) {
    Print(L"Found no EFI_USB_IO handles after connecting controllers (status=%r)\n", Status);
  } else {
    Print(L"Found %u EFI_USB_IO handles\n\n", (UINT32)UsbCount);

    // For each PCI host, map child USB handles by device-path prefix
    for (UINTN i = 0; i < PciCount; ++i) {
      EFI_PCI_IO_PROTOCOL *PciIo = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
      if (EFI_ERROR(Status) || PciIo == NULL) continue;

      // check class again
      UINT32 ClassReg = 0;
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassReg);
      if (EFI_ERROR(Status)) continue;
      UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
      UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
      if (!(BaseClass == 0x0C && SubClass == 0x03)) continue;

      EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
      UINTN PciDpSize = (PciDp != NULL) ? GetDevicePathSizeBytes(PciDp) : 0;
      Print(L"Mapping USB devices to PCI host handle=0x%p\n", PciHandles[i]);

      for (UINTN u = 0; u < UsbCount; ++u) {
        EFI_DEVICE_PATH_PROTOCOL *UsbDp = NULL;
        Status = gBS->HandleProtocol(UsbHandles[u], &gEfiDevicePathProtocolGuid, (VOID **)&UsbDp);
        if (EFI_ERROR(Status) || UsbDp == NULL) continue;

        UINTN UsbDpSize = GetDevicePathSizeBytes(UsbDp);
        if (PciDp != NULL && UsbDpSize >= PciDpSize && DevicePathIsPrefixBinary(PciDp, PciDpSize, UsbDp, UsbDpSize)) {
          // child of this host
          EFI_USB_IO_PROTOCOL *UsbIo = NULL;
          Status = gBS->HandleProtocol(UsbHandles[u], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
          if (EFI_ERROR(Status) || UsbIo == NULL) continue;

          EFI_USB_DEVICE_DESCRIPTOR DevDesc;
          ZeroMem(&DevDesc, sizeof(DevDesc));
          Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
          if (EFI_ERROR(Status)) {
            Print(L"  USB handle=0x%p: UsbGetDeviceDescriptor failed: %r\n", UsbHandles[u], Status);
          } else {
            // Print device info and textual device path if possible
            CHAR16 *Txt = NULL;
            if (DpToText != NULL) {
              Txt = DpToText->ConvertDevicePathToText(UsbDp, FALSE, FALSE);
            }
            Print(L"  USB Device: Handle=0x%p  VID=0x%04x PID=0x%04x Class=0x%02x\n",
                  UsbHandles[u], DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
            if (Txt) {
              Print(L"    DevicePath: %s\n", Txt);
              FreePool(Txt);
            }
          }
        }
      } // for usb handles

      Print(L"\n");
    } // for pci hosts
  }

  // cleanup
  if (PciHandles) FreePool(PciHandles);
  if (UsbHandles) FreePool(UsbHandles);

  Print(L"Done. Press any key to exit...\n");
  // wait for key press
  {
    EFI_INPUT_KEY Key;
    while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_NOT_READY) {
      // spin
    }
  }

  return EFI_SUCCESS;
}
