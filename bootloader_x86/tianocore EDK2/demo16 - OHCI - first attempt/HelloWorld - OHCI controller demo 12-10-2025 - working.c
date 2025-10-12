/** UsbPciEnumerate.c
 *
 *  - Enumerate PCI devices and find USB host controllers (class 0x0C / subclass 0x03)
 *  - Print PCI info (Vendor/Device, BARs, IRQ)
 *  - For OHCI (ProgIf == 0x10) read some OHCI MMIO registers from BAR0
 *  - Connect controller, find matching EFI_USB2_HC_PROTOCOL instance, show ports
 *  - Re-scan EFI_USB_IO handles and map them to root-hub ports (by parsing USB() device-path nodes)
 *
 */

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/PrintLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/MemoryAllocationLib.h>

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

#define MESSAGING_DEVICE_PATH 0x03
#define MSG_USB               0x05

//
// Helpers
//

STATIC
UINTN
GetDevicePathSizeBytes (
  IN EFI_DEVICE_PATH_PROTOCOL *DevicePath
  )
{
  if (DevicePath == NULL) return 0;
  EFI_DEVICE_PATH_PROTOCOL *Node = DevicePath;
  UINTN Size = 0;
  for (;;) {
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

// Find an EFI_USB2_HC_PROTOCOL instance that is under the given PCI device path.
// We locate all handles that support EFI_USB2_HC_PROTOCOL and pick the one whose
// device path is a child of the PCI device path (binary prefix match).
STATIC
EFI_USB2_HC_PROTOCOL *
FindUsb2HcForPciHost (
  IN EFI_DEVICE_PATH_PROTOCOL *PciDp,
  IN UINTN                    PciDpSize
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;
  EFI_USB2_HC_PROTOCOL *Usb2 = NULL;

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &Count, &Handles);
  if (EFI_ERROR(Status) || Count == 0) return NULL;

  for (UINTN i = 0; i < Count; ++i) {
    EFI_DEVICE_PATH_PROTOCOL *Dp = NULL;
    Status = gBS->HandleProtocol(Handles[i], &gEfiDevicePathProtocolGuid, (VOID **)&Dp);
    if (EFI_ERROR(Status) || Dp == NULL) continue;
    UINTN DpSize = GetDevicePathSizeBytes(Dp);
    if (PciDp != NULL && DpSize >= PciDpSize && DevicePathIsPrefixBinary(PciDp, PciDpSize, Dp, DpSize)) {
      // This Usb2Hc belongs under the PCI host
      Status = gBS->HandleProtocol(Handles[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Usb2);
      if (!EFI_ERROR(Status) && Usb2 != NULL) {
        break;
      }
    }
  }

  if (Handles) FreePool(Handles);
  return Usb2;
}

// Parse binary device path nodes to extract all USB() nodes' (ParentPortNumber, InterfaceNumber).
// Caller must FreePool ports/ifaces arrays.
STATIC
EFI_STATUS
ExtractUsbNodesFromDevicePath(
  IN  EFI_DEVICE_PATH_PROTOCOL *DevPath,
  OUT UINT32                  **Ports,
  OUT UINT32                  **Ifaces,
  OUT UINTN                   *Count
  )
{
  *Ports = NULL;
  *Ifaces = NULL;
  *Count = 0;
  if (DevPath == NULL) return EFI_SUCCESS;

  EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
  UINTN total = 0;

  while (TRUE) {
    UINT8 *Raw = (UINT8 *)Node;
    UINT16 Len = *(UINT16 *)(Raw + 2);
    if (Len < sizeof(EFI_DEVICE_PATH_PROTOCOL)) break;

    if (Node->Type == MESSAGING_DEVICE_PATH && Node->SubType == MSG_USB && Len >= (sizeof(EFI_DEVICE_PATH_PROTOCOL) + 2)) {
      UINT8 parentPort = Raw[4];
      UINT8 iface = Raw[5];
      UINT32 *newPorts = ReallocatePool(total * sizeof(UINT32), (total + 1) * sizeof(UINT32), *Ports);
      UINT32 *newIfaces = ReallocatePool(total * sizeof(UINT32), (total + 1) * sizeof(UINT32), *Ifaces);
      if (newPorts == NULL || newIfaces == NULL) {
        if (newPorts) FreePool(newPorts);
        if (newIfaces) FreePool(newIfaces);
        return EFI_OUT_OF_RESOURCES;
      }
      *Ports = newPorts;
      *Ifaces = newIfaces;
      (*Ports)[total] = (UINT32)parentPort;
      (*Ifaces)[total] = (UINT32)iface;
      total++;
    }

    if (Node->Type == END_DEVICE_PATH_TYPE && Node->SubType == END_ENTIRE_DEVICE_PATH_SUBTYPE) break;
    Node = (EFI_DEVICE_PATH_PROTOCOL *)((UINT8 *)Node + Len);
  }

  *Count = total;
  return EFI_SUCCESS;
}

// Find first UsbIo handle which is child of PciDevicePath and contains a USB() node with PortNumber.
STATIC
EFI_HANDLE
FindUsbHandleForPort(
  IN EFI_HANDLE                *UsbHandles,
  IN UINTN                     UsbCount,
  IN EFI_DEVICE_PATH_PROTOCOL *PciDp,
  IN UINTN                     PciDpSize,
  IN UINT32                    PortNumber
  )
{
  for (UINTN u = 0; u < UsbCount; ++u) {
    EFI_DEVICE_PATH_PROTOCOL *UsbDp = NULL;
    EFI_STATUS Status = gBS->HandleProtocol(UsbHandles[u], &gEfiDevicePathProtocolGuid, (VOID **)&UsbDp);
    if (EFI_ERROR(Status) || UsbDp == NULL) continue;

    UINTN UsbDpSize = GetDevicePathSizeBytes(UsbDp);
    if (PciDp != NULL && (UsbDpSize < PciDpSize || !DevicePathIsPrefixBinary(PciDp, PciDpSize, UsbDp, UsbDpSize))) {
      continue;
    }

    UINT32 *ports = NULL;
    UINT32 *ifaces = NULL;
    UINTN count = 0;
    Status = ExtractUsbNodesFromDevicePath(UsbDp, &ports, &ifaces, &count);
    if (EFI_ERROR(Status)) {
      if (ports) FreePool(ports);
      if (ifaces) FreePool(ifaces);
      continue;
    }
    BOOLEAN matched = FALSE;
    for (UINTN k = 0; k < count; ++k) {
      if (ports[k] == PortNumber) { matched = TRUE; break; }
    }
    if (ports) FreePool(ports);
    if (ifaces) FreePool(ifaces);
    if (matched) {
      return UsbHandles[u];
    }
  }
  return NULL;
}

//
// Main
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
  UINTN PciCount = 0;

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Print(L"No PCI devices found (status=%r)\n", Status);
    return EFI_SUCCESS;
  }

  // Optionally print DevicePath->Text for nicer outputs
  EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpToText = NULL;
  gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID **)&DpToText);

  Print(L"Scanning %u PCI devices for USB host controllers...\n\n", (UINT32)PciCount);

  // Iterate PCI handles
  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) continue;

    // read class/prog-if at offset 0x08 (DWORD)
    UINT32 ClassReg = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)0x08, 1, &ClassReg);
    if (EFI_ERROR(Status)) continue;
    UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
    UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
    UINT8 ProgIf    = (ClassReg >> 8)  & 0xFF;

    if (!(BaseClass == 0x0C && SubClass == 0x03)) continue; // not USB host controller

    Print(L"PCI USB Host Controller found (handle=0x%p): Class=0x%02x Sub=0x%02x ProgIf=0x%02x\n",
          PciHandles[i], (UINT32)BaseClass, (UINT32)SubClass, (UINT32)ProgIf);

    // show PCI location if available
    if (PciIo->GetLocation != NULL) {
      UINTN Seg, Bus, Dev, Func;
      if (!EFI_ERROR(PciIo->GetLocation(PciIo, &Seg, &Bus, &Dev, &Func))) {
        Print(L"  PCI Location: Seg=%u Bus=%u Dev=%u Func=%u\n", Seg, Bus, Dev, Func);
      }
    }

    // Read vendor/device
    UINT32 Vd = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)0x00, 1, &Vd);
    if (!EFI_ERROR(Status)) {
      UINT16 Vid = (UINT16)(Vd & 0xFFFF);
      UINT16 Did = (UINT16)((Vd >> 16) & 0xFFFF);
      Print(L"  PCI VendorId=0x%04x DeviceId=0x%04x\n", Vid, Did);
    }

    // Read BARs 0x10..0x24
    for (UINTN barOff = 0x10; barOff <= 0x24; barOff += 4) {
      UINT32 BarVal = 0;
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)barOff, 1, &BarVal);
      if (!EFI_ERROR(Status)) {
        Print(L"  BAR 0x%02x = 0x%08x\n", (UINT32)barOff, BarVal);
      }
    }

    // Interrupt line/pin at 0x3C/0x3D
    UINT8 IntLine = 0, IntPin = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint8, (UINT32)0x3C, 1, &IntLine);
    if (!EFI_ERROR(Status)) {
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint8, (UINT32)0x3D, 1, &IntPin);
      Print(L"  Interrupt Line=0x%02x  Pin=0x%02x\n", IntLine, IntPin);
    }

    // Show textual device path if available
    EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
    if (!EFI_ERROR(Status) && PciDp != NULL && DpToText != NULL) {
      CHAR16 *Txt = DpToText->ConvertDevicePathToText(PciDp, FALSE, FALSE);
      if (Txt) { Print(L"  PCI DevicePath: %s\n", Txt); FreePool(Txt); }
    }

    //
    // OHCI specifics: if ProgIf == 0x10 (OHCI), try to read some registers from BAR0
    //
    if (ProgIf == 0x10) {
      Print(L"  Detected OHCI host controller (ProgIf=0x10). Attempting to read OHCI MMIO registers from BAR0...\n");
      // Try to read HcRevision (offset 0x00), HcControl (0x04), HcCommandStatus (0x08), HcHCCA (0x18)
      UINT32 val;
      Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0 /* BAR0 index */, 0x00, 1, &val);
      if (!EFI_ERROR(Status)) Print(L"    HcRevision(0x00) = 0x%08x\n", val);
      else Print(L"    HcRevision read failed: %r\n", Status);

      Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0, 0x04, 1, &val);
      if (!EFI_ERROR(Status)) Print(L"    HcControl(0x04) = 0x%08x\n", val);

      Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0, 0x08, 1, &val);
      if (!EFI_ERROR(Status)) Print(L"    HcCommandStatus(0x08) = 0x%08x\n", val);

      Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0, 0x18, 1, &val);
      if (!EFI_ERROR(Status)) Print(L"    HcHCCA(0x18) = 0x%08x\n", val);
    }

    //
    // Connect controller so bus driver can bind and create root-hub and children
    //
    Status = gBS->ConnectController(PciHandles[i], NULL, NULL, TRUE);
    if (EFI_ERROR(Status)) {
      Print(L"  ConnectController returned: %r\n", Status);
    } else {
      Print(L"  ConnectController: OK\n");
    }

    //
    // Find the Usb2Hc for this PCI host (may be installed on the PCI handle or child handles)
    //
    UINTN PciDpSize = (PciDp != NULL) ? GetDevicePathSizeBytes(PciDp) : 0;
    EFI_USB2_HC_PROTOCOL *Usb2Hc = FindUsb2HcForPciHost(PciDp, PciDpSize);
    if (Usb2Hc == NULL) {
      // try direct handle protocol as a fallback
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Usb2Hc);
    }

    if (Usb2Hc == NULL) {
      Print(L"  EFI_USB2_HC_PROTOCOL not present for this host (or not found under PCI subtree).\n\n");
      continue;
    }

    // Query host capabilities
    UINT8 maxSpeed = 0, numPorts = 0, is64 = 0;
    Status = Usb2Hc->GetCapability(Usb2Hc, &maxSpeed, &numPorts, &is64);
    if (EFI_ERROR(Status)) {
      Print(L"  Usb2Hc->GetCapability failed: %r\n\n", Status);
      continue;
    }
    Print(L"  Host Capabilities: maxSpeed=%u numPorts=%u 64bit=%u\n", (UINT32)maxSpeed, (UINT32)numPorts, (UINT32)is64);

    // Re-scan USB_IO handles after attempting ConnectController
    EFI_HANDLE *UsbHandles = NULL;
    UINTN UsbCount = 0;
    Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
    if (EFI_ERROR(Status) || UsbCount == 0) {
      Print(L"  No EFI_USB_IO handles found yet (status=%r). You may need to reset ports or ensure drivers are present.\n\n", Status);
      if (UsbHandles) { FreePool(UsbHandles); UsbHandles = NULL; }
      continue;
    }

    // For each root hub port, query status; if connected, try to map to a USB_IO handle
    for (UINT8 port = 1; port <= numPorts; ++port) {
      EFI_USB_PORT_STATUS PortStatus;
      Status = Usb2Hc->GetRootHubPortStatus(Usb2Hc, port, &PortStatus);
      if (EFI_ERROR(Status)) {
        Print(L"    Port %u: GetRootHubPortStatus failed: %r\n", (UINT32)port, Status);
        continue;
      }
      BOOLEAN connected = (PortStatus.PortStatus & USB_PORT_STAT_CONNECTION) != 0;
      Print(L"    Port %u: Connected=%u  PortStatus=0x%08x\n", (UINT32)port, connected ? 1 : 0, (UINT32)PortStatus.PortStatus);

      if (!connected) continue;

      // Optionally force reset to trigger enumeration (uncomment if you want)
      // Usb2Hc->SetRootHubPortFeature(Usb2Hc, port, EfiUsbPortReset);
      // gBS->Stall(200000); // 200 ms for reset handling

      // Try to find a matching USB handle for this port
      EFI_HANDLE matched = FindUsbHandleForPort(UsbHandles, UsbCount, PciDp, PciDpSize, (UINT32)port);
      if (matched == NULL) {
        Print(L"      No matching EFI_USB_IO handle found for Port %u (consider resetting port and re-scanning)\n", (UINT32)port);
        continue;
      }

      // Get device descriptor from matched handle
      EFI_USB_IO_PROTOCOL *UsbIo = NULL;
      Status = gBS->HandleProtocol(matched, &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
      if (EFI_ERROR(Status) || UsbIo == NULL) {
        Print(L"      Matched handle 0x%p but cannot access EFI_USB_IO_PROTOCOL: %r\n", matched, Status);
        continue;
      }

      EFI_USB_DEVICE_DESCRIPTOR DevDesc;
      ZeroMem(&DevDesc, sizeof(DevDesc));
      Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
      if (EFI_ERROR(Status)) {
        Print(L"      UsbGetDeviceDescriptor failed: %r\n", Status);
      } else {
        Print(L"      Device Handle=0x%p  VID=0x%04x PID=0x%04x Class=0x%02x\n",
              matched, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
      }

      // Optionally show textual device path
      if (DpToText != NULL) {
        EFI_DEVICE_PATH_PROTOCOL *MatchedDp = NULL;
        Status = gBS->HandleProtocol(matched, &gEfiDevicePathProtocolGuid, (VOID **)&MatchedDp);
        if (!EFI_ERROR(Status) && MatchedDp != NULL) {
          CHAR16 *dpText = DpToText->ConvertDevicePathToText(MatchedDp, FALSE, FALSE);
          if (dpText) { Print(L"        DevicePath: %s\n", dpText); FreePool(dpText); }
        }
      }
    } // port loop

    if (UsbHandles) FreePool(UsbHandles);

    Print(L"\n");
  } // pci loop

  if (PciHandles) FreePool(PciHandles);

  Print(L"Done. Press any key to exit...\n");
  {
    EFI_INPUT_KEY Key;
    while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_NOT_READY) {
      // simply spin until key pressed
    }
  }

  return EFI_SUCCESS;
}
