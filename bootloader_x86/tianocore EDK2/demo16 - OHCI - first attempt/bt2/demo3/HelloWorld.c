//
// UsbHostDeepScan.c
// - Connects PCI USB host controllers, queries root-hub ports via EFI_USB2_HC_PROTOCOL,
// - Enumerates EFI_USB_IO_PROTOCOL handles and maps them to the host (device-path prefix).
//

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/PrintLib.h>
#include <Protocol/PciIo.h>
#include <Protocol/Usb2HostController.h>
#include <Protocol/DevicePath.h>
#include <Protocol/DevicePathToText.h>
#include <Protocol/UsbIo.h>
#include <IndustryStandard/Usb.h>

#ifndef END_DEVICE_PATH_TYPE
#define END_DEVICE_PATH_TYPE 0x7F
#endif
#ifndef END_ENTIRE_DEVICE_PATH_SUBTYPE
#define END_ENTIRE_DEVICE_PATH_SUBTYPE 0xFF
#endif

#ifndef USB_PORT_STAT_CONNECTION
#define USB_PORT_STAT_CONNECTION 0x00000001
#endif


STATIC
UINTN
GetDevicePathSizeBytes (
  IN EFI_DEVICE_PATH_PROTOCOL  *DevicePath
  )
{
  EFI_DEVICE_PATH_PROTOCOL *Node;
  UINTN Size = 0;
  if (DevicePath == NULL) return 0;
  Node = DevicePath;
  while (TRUE) {
    UINT8 *Raw = (UINT8 *)Node;
    UINT16 Len = *(UINT16 *)(Raw + 2);
    if (Len < sizeof(EFI_DEVICE_PATH_PROTOCOL)) return Size;
    Size += Len;
    if (Node->Type == END_DEVICE_PATH_TYPE && Node->SubType == END_ENTIRE_DEVICE_PATH_SUBTYPE) break;
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
  if (Prefix == NULL || Path == NULL) return FALSE;
  if (PrefixSize == 0 || PathSize < PrefixSize) return FALSE;
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

  // Locate all PCI handles that implement EFI_PCI_IO_PROTOCOL
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Print(L"No PCI devices / error %r\n", Status);
    return EFI_SUCCESS;
  }

  // Optional: DevicePath->Text for nicer output
  EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpToText = NULL;
  gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID**)&DpToText);

  Print(L"Scanning %u PCI devices for USB host controllers...\n\n", (UINT32)PciCount);

  // We'll reconnect controllers to ensure USB stack creates root-hub and child devices.
  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) {
      continue;
    }

    // Read class code/reg at offset 0x08 (DWORD): [BaseClass:SubClass:ProgIf:Revision]
    UINT32 ClassReg = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassReg);
    if (EFI_ERROR(Status)) continue;

    UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
    UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
    UINT8 ProgIf    = (ClassReg >> 8)  & 0xFF;

    // Check for Serial Bus (0x0C) / USB (0x03)
    if (BaseClass == 0x0C && SubClass == 0x03) {
      Print(L"PCI USB Host Controller found (handle=0x%p): Class=0x%02x Sub=0x%02x ProgIf=0x%02x\n",
            PciHandles[i], (UINT32)BaseClass, (UINT32)SubClass, (UINT32)ProgIf);

      // show device path if possible
      EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
      if (!EFI_ERROR(Status) && PciDp != NULL && DpToText != NULL) {
        CHAR16 *Txt = DpToText->ConvertDevicePathToText(PciDp, FALSE, FALSE);
        if (Txt) {
          Print(L"  PCI DevicePath: %s\n", Txt);
          FreePool(Txt);
        }
      }

      // Force a connect on the controller to let the USB bus driver enumerate the root hub and children.
      // This is important: without ConnectController the system may not have created the USB_IO handles.
      Status = gBS->ConnectController(PciHandles[i], NULL, NULL, TRUE);
      if (EFI_ERROR(Status)) {
        Print(L"  ConnectController returned: %r\n", Status);
      } else {
        Print(L"  ConnectController: OK\n");
      }

      // Attempt to get EFI_USB2_HC_PROTOCOL from the same handle (some implementations install it here)
      EFI_USB2_HC_PROTOCOL *Usb2Hc = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Usb2Hc);
      if (!EFI_ERROR(Status) && Usb2Hc != NULL) {
        UINT8 maxSpeed = 0;
        UINT8 numPorts = 0;
        UINT8 is64 = 0;
        Status = Usb2Hc->GetCapability(Usb2Hc, &maxSpeed, &numPorts, &is64);
        if (!EFI_ERROR(Status)) {
          Print(L"  Host Capabilities: maxSpeed=%u numPorts=%u 64bit=%u\n", (UINT32)maxSpeed, (UINT32)numPorts, (UINT32)is64);
          
		  for (UINT8 port = 1; port <= numPorts; ++port) {
			EFI_USB_PORT_STATUS PortStatus;
			Status = Usb2Hc->GetRootHubPortStatus(Usb2Hc, port, &PortStatus);
			if (EFI_ERROR(Status)) {
				Print(L"    Port %u: GetRootHubPortStatus failed: %r\n", (UINT32)port, Status);
			} else {
				// Some EFI implementations only expose PortStatus (no PortChange member).
				// Use the standard connection bit to detect device presence.
				BOOLEAN connected = (PortStatus.PortStatus & USB_PORT_STAT_CONNECTION) != 0;
				Print(L"    Port %u: Connected=%u  PortStatus=0x%08x\n",
					  (UINT32)port, connected ? 1 : 0, (UINT32)PortStatus.PortStatus);
			}
		}

				  
		  /*
		  for (UINT8 port = 1; port <= numPorts; ++port) {
            EFI_USB_PORT_STATUS PortStatus;
            Status = Usb2Hc->GetRootHubPortStatus(Usb2Hc, port, &PortStatus);
            if (EFI_ERROR(Status)) {
              Print(L"    Port %u: GetRootHubPortStatus failed: %r\n", (UINT32)port, Status);
            } else {
              // PortStatus.PortStatus bits follow USB spec (bit 0 = current connect status etc.)
              BOOLEAN connected = (PortStatus.PortStatus & USB_PORT_STAT_CONNECTION) != 0;
              UINT32 portSpeed = (PortStatus.PortStatus & USB_PORT_STAT_SPEED_MASK); // may be controller-specific
              Print(L"    Port %u: Connected=%u  PortStatus=0x%08x  PortChange=0x%08x\n",
                    (UINT32)port, connected ? 1 : 0, (UINT32)PortStatus.PortStatus, (UINT32)PortStatus.PortChange);
            }
          }
		  */
        } else {
          Print(L"  Usb2Hc->GetCapability failed: %r\n", Status);
        }
      } else {
        Print(L"  EFI_USB2_HC_PROTOCOL not present on this handle (status=%r)\n", Status);
      }

      Print(L"\n");
    } // if USB host
  } // for each pci handle

  //
  // After trying to connect controllers, enumerate all EFI_USB_IO_PROTOCOL handles
  // and map them to the PCI host controllers by device-path prefix.
  //
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
  if (EFI_ERROR(Status) || UsbCount == 0) {
    Print(L"No USB_IO handles found after connect: %r\n", Status);
  } else {
    Print(L"Found %u USB_IO handles after connecting controllers\n\n", (UINT32)UsbCount);

    // Build list of PCI DP pointers and sizes for mapping
    // Re-fetch PCI handles to ensure we still have list
    // (we can reuse PciHandles/PciCount from above)
    for (UINTN i = 0; i < PciCount; ++i) {
      // For each PCI host, compute dp size once
      EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
      UINTN PciDpSize = (PciDp != NULL) ? GetDevicePathSizeBytes(PciDp) : 0;

      // Only consider those that are USB host controllers (class check again)
      EFI_PCI_IO_PROTOCOL *PciIo = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
      if (EFI_ERROR(Status) || PciIo == NULL) continue;
      UINT32 ClassReg = 0;
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassReg);
      if (EFI_ERROR(Status)) continue;
      UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
      UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
      if (!(BaseClass == 0x0C && SubClass == 0x03)) continue;

      Print(L"Mapping devices to PCI host (handle=0x%p)...\n", PciHandles[i]);

      for (UINTN u = 0; u < UsbCount; ++u) {
        EFI_DEVICE_PATH_PROTOCOL *UsbDp = NULL;
        Status = gBS->HandleProtocol(UsbHandles[u], &gEfiDevicePathProtocolGuid, (VOID **)&UsbDp);
        if (EFI_ERROR(Status) || UsbDp == NULL) continue;
        UINTN UsbDpSize = GetDevicePathSizeBytes(UsbDp);

        if (PciDp != NULL && UsbDpSize >= PciDpSize && DevicePathIsPrefixBinary(PciDp, PciDpSize, UsbDp, UsbDpSize)) {
          // This USB handle is a child of the PCI host
          EFI_USB_IO_PROTOCOL *UsbIo = NULL;
          Status = gBS->HandleProtocol(UsbHandles[u], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
          if (EFI_ERROR(Status) || UsbIo == NULL) continue;

          // Get Device Descriptor
          EFI_USB_DEVICE_DESCRIPTOR DevDesc;
          ZeroMem(&DevDesc, sizeof(DevDesc));
          Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
          if (EFI_ERROR(Status)) {
            Print(L"  USB handle=0x%p: UsbGetDeviceDescriptor failed: %r\n", UsbHandles[u], Status);
          } else {
            // print basic info
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

            // You can optionally attempt to parse configuration/interface descriptors here
            // using UsbIo->UsbGetConfigDescriptor or by control-transfer GET_DESCRIPTOR
            // (omitted here to keep this demo robust across implementations).
          }
        }
      } // usb handles
      Print(L"\n");
    } // for each pci handle
  }

  // cleanup
  if (PciHandles) FreePool(PciHandles);
  if (UsbHandles) FreePool(UsbHandles);

  Print(L"Done. Press any key to exit...\n");
  {
    EFI_INPUT_KEY Key;
    while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_NOT_READY) {
      // spin
    }
  }

  return EFI_SUCCESS;
}
