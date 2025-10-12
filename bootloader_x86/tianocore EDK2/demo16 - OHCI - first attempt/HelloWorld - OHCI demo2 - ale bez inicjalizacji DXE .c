/** UsbPciDiagnostic.c
 *
 * Diagnostic UEFI application to enumerate USB hosts and OHCI specifics.
 *
 * Build in EDK2 (MdeModulePkg/Application).
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

// --------- Helpers ---------

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

STATIC
VOID
DumpProtocolsOnHandle(
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;
  EFI_GUID **ProtocolBuffer = NULL;
  UINTN ProtocolCount = 0;

  Status = gBS->ProtocolsPerHandle(Handle, &ProtocolBuffer, &ProtocolCount);
  if (EFI_ERROR(Status)) {
    Print(L"    ProtocolsPerHandle failed: %r\n", Status);
    return;
  }

  for (UINTN i = 0; i < ProtocolCount; ++i) {
    Print(L"    Protocol[%u]: %g\n", (UINT32)i, &ProtocolBuffer[i]);
  }

  if (ProtocolBuffer) {
    gBS->FreePool(ProtocolBuffer);
  }
}

// Find any EFI_USB2_HC_PROTOCOL handle whose device path is a child of the given PCI DP.
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
  if (!EFI_ERROR(Status) && Count > 0 && Handles != NULL) {
    for (UINTN i = 0; i < Count; ++i) {
      EFI_DEVICE_PATH_PROTOCOL *Dp = NULL;
      Status = gBS->HandleProtocol(Handles[i], &gEfiDevicePathProtocolGuid, (VOID **)&Dp);
      if (EFI_ERROR(Status) || Dp == NULL) continue;
      UINTN DpSize = GetDevicePathSizeBytes(Dp);
      if (PciDp != NULL && DpSize >= PciDpSize && DevicePathIsPrefixBinary(PciDp, PciDpSize, Dp, DpSize)) {
        Status = gBS->HandleProtocol(Handles[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Usb2);
        if (!EFI_ERROR(Status) && Usb2 != NULL) break;
      }
    }
  }
  if (Handles) FreePool(Handles);
  return Usb2;
}

// Parse device path and extract all (ParentPort, Interface) for USB() nodes
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

// Diagnostic helper: print any Usb2Hc handles and their device-paths
STATIC
VOID
ListAllUsb2HcHandles(VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &Count, &Handles);
  if (EFI_ERROR(Status) || Count == 0) {
    Print(L"No EFI_USB2_HC_PROTOCOL handles found: %r\n", Status);
    return;
  }
  Print(L"Found %u EFI_USB2_HC_PROTOCOL handle(s):\n", (UINT32)Count);
  EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpToText = NULL;
  gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID **)&DpToText);

  for (UINTN i = 0; i < Count; ++i) {
    Print(L"  Usb2Hc handle=0x%p\n", Handles[i]);
    if (DpToText != NULL) {
      EFI_DEVICE_PATH_PROTOCOL *Dp = NULL;
      Status = gBS->HandleProtocol(Handles[i], &gEfiDevicePathProtocolGuid, (VOID **)&Dp);
      if (!EFI_ERROR(Status) && Dp != NULL) {
        CHAR16 *txt = DpToText->ConvertDevicePathToText(Dp, FALSE, FALSE);
        if (txt) { Print(L"    DevicePath: %s\n", txt); FreePool(txt); }
      }
    }
    DumpProtocolsOnHandle(Handles[i]);
  }
  FreePool(Handles);
}

// --------- Main ---------

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

  // Optionally DevicePath->Text
  EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpToText = NULL;
  gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID **)&DpToText);

  Print(L"Scanning %u PCI devices for USB host controllers...\n\n", (UINT32)PciCount);

  // Print all Usb2Hc handles (global)
  ListAllUsb2HcHandles();
  Print(L"\n");

  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) continue;

    // read class/prog-if at offset 0x08
    UINT32 ClassReg = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)0x08, 1, &ClassReg);
    if (EFI_ERROR(Status)) continue;
    UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
    UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
    UINT8 ProgIf    = (ClassReg >> 8)  & 0xFF;

    if (!(BaseClass == 0x0C && SubClass == 0x03)) continue; // not USB host controller

    Print(L"PCI USB Host Controller found (handle=0x%p): Class=0x%02x Sub=0x%02x ProgIf=0x%02x\n",
          PciHandles[i], (UINT32)BaseClass, (UINT32)SubClass, (UINT32)ProgIf);

    // location
    if (PciIo->GetLocation != NULL) {
      UINTN Seg, Bus, Dev, Func;
      if (!EFI_ERROR(PciIo->GetLocation(PciIo, &Seg, &Bus, &Dev, &Func))) {
        Print(L"  PCI Location: Seg=%u Bus=%u Dev=%u Func=%u\n", Seg, Bus, Dev, Func);
      }
    }

    // vendor/device
    UINT32 Vd = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)0x00, 1, &Vd);
    if (!EFI_ERROR(Status)) {
      UINT16 Vid = (UINT16)(Vd & 0xFFFF);
      UINT16 Did = (UINT16)((Vd >> 16) & 0xFFFF);
      Print(L"  PCI VendorId=0x%04x DeviceId=0x%04x\n", Vid, Did);
    }

    // BARs 0x10..0x24
    for (UINTN barOff = 0x10; barOff <= 0x24; barOff += 4) {
      UINT32 BarVal = 0;
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)barOff, 1, &BarVal);
      if (!EFI_ERROR(Status)) {
        Print(L"  BAR 0x%02x = 0x%08x\n", (UINT32)barOff, BarVal);
      }
    }

    // PCI command register
    UINT16 Cmd = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint16, (UINT32)0x04, 1, &Cmd);
    if (!EFI_ERROR(Status)) {
      Print(L"  PCI Command: 0x%04x (MemSpace=%u IO=%u BusMaster=%u)\n",
            Cmd, (Cmd & 0x2) ? 1:0, (Cmd & 0x1) ? 1:0, (Cmd & 0x4) ? 1:0);
    }

    // IRQ line/pin
    UINT8 IntLine = 0, IntPin = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint8, (UINT32)0x3C, 1, &IntLine);
    if (!EFI_ERROR(Status)) {
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint8, (UINT32)0x3D, 1, &IntPin);
      Print(L"  Interrupt Line=0x%02x  Pin=0x%02x\n", IntLine, IntPin);
    }

    // show PCI device path
    EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
    if (!EFI_ERROR(Status) && PciDp != NULL && DpToText != NULL) {
      CHAR16 *Txt = DpToText->ConvertDevicePathToText(PciDp, FALSE, FALSE);
      if (Txt) { Print(L"  PCI DevicePath: %s\n", Txt); FreePool(Txt); }
    }

    // OHCI-specific: ProgIf == 0x10
    if (ProgIf == 0x10) {
      Print(L"  Detected OHCI (ProgIf=0x10). Checking BAR0 MMIO and OHCI registers...\n");
      // Read BAR0
      UINT32 Bar0 = 0;
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, (UINT32)0x10, 1, &Bar0);
      if (!EFI_ERROR(Status)) {
        if (Bar0 & 0x1) {
          Print(L"    BAR0 indicates IO space (not MMIO). BAR0=0x%08x\n", Bar0);
        } else {
          UINT64 base = (UINT64)(Bar0 & ~0xFULL);
          Print(L"    BAR0 MMIO base = 0x%lx\n", base);
          // Try to read OHCI regs via PciIo->Mem.Read (BAR index 0)
          UINT32 val = 0;
          Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0 /*BarIndex*/, 0x00, 1, &val);
          if (!EFI_ERROR(Status)) Print(L"    HcRevision(0x00) = 0x%08x\n", val); else Print(L"    HcRevision read failed: %r\n", Status);
          Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0, 0x04, 1, &val);
          if (!EFI_ERROR(Status)) Print(L"    HcControl(0x04) = 0x%08x\n", val);
          Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0, 0x08, 1, &val);
          if (!EFI_ERROR(Status)) Print(L"    HcCommandStatus(0x08) = 0x%08x\n", val);
          Status = PciIo->Mem.Read(PciIo, EfiPciIoWidthUint32, 0, 0x18, 1, &val);
          if (!EFI_ERROR(Status)) Print(L"    HcHCCA(0x18) = 0x%08x\n", val);
          // HcHCCA==0 likely means driver hasn't allocated/installed HCCA (controller not started).
        }
      }
    } // OHCI

    // Try to connect controller (bind driver)
    Status = gBS->ConnectController(PciHandles[i], NULL, NULL, TRUE);
    if (EFI_ERROR(Status)) {
      Print(L"  ConnectController returned: %r\n", Status);
    } else {
      Print(L"  ConnectController: OK\n");
    }

    // After ConnectController try to find Usb2Hc for this host
    UINTN PciDpSize = (PciDp != NULL) ? GetDevicePathSizeBytes(PciDp) : 0;
    EFI_USB2_HC_PROTOCOL *Usb2Hc = FindUsb2HcForPciHost(PciDp, PciDpSize);
    if (Usb2Hc == NULL) {
      // fallback to direct protocol on the PCI handle
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Usb2Hc);
    }

    if (Usb2Hc == NULL) {
      Print(L"  EFI_USB2_HC_PROTOCOL not present for this host (driver likely missing or not bound).\n");
      Print(L"  Suggestion: run on firmware image with OHCI/EHCI/xHCI DXE driver or add the appropriate DXE driver.\n\n");
      continue;
    }

    // USB2_HC present -> query capabilities and ports
    UINT8 maxSpeed = 0, numPorts = 0, is64 = 0;
    Status = Usb2Hc->GetCapability(Usb2Hc, &maxSpeed, &numPorts, &is64);
    if (EFI_ERROR(Status)) {
      Print(L"  Usb2Hc->GetCapability failed: %r\n\n", Status);
      continue;
    }
    Print(L"  Host Capabilities: maxSpeed=%u numPorts=%u 64bit=%u\n", (UINT32)maxSpeed, (UINT32)numPorts, (UINT32)is64);

    // Rescan EFI_USB_IO handles after connect
    EFI_HANDLE *UsbHandles = NULL;
    UINTN UsbCount = 0;
    Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
    if (EFI_ERROR(Status)) { UsbCount = 0; UsbHandles = NULL; }

    // Query each port
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

      // Optionally force reset to trigger enumeration (commented: use only when safe)
      // Print(L"      Attempting port reset (CAREFUL: this disconnects the device briefly)...\n");
      // Usb2Hc->SetRootHubPortFeature(Usb2Hc, (UINT8)port, EfiUsbPortReset);
      // gBS->Stall(300000); // 300 ms

      // Re-locate UsbIo handles if you did a reset. For now we use previously scanned UsbHandles
      if (UsbCount > 0 && UsbHandles != NULL) {
        EFI_HANDLE matched = FindUsbHandleForPort(UsbHandles, UsbCount, PciDp, PciDpSize, (UINT32)port);
        if (matched == NULL) {
          Print(L"      No matching EFI_USB_IO handle found for Port %u (try resetting port and re-scan)\n", (UINT32)port);
          continue;
        }

        EFI_USB_IO_PROTOCOL *UsbIo = NULL;
        Status = gBS->HandleProtocol(matched, &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
        if (EFI_ERROR(Status) || UsbIo == NULL) {
          Print(L"      Matched handle=0x%p but cannot access EFI_USB_IO_PROTOCOL: %r\n", matched, Status);
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
      } // if UsbHandles
    } // port loop

    if (UsbHandles) FreePool(UsbHandles);

    Print(L"\n");
  } // for each PCI handle

  if (PciHandles) FreePool(PciHandles);

  Print(L"Diagnostics complete. Press any key to exit...\n");
  {
    EFI_INPUT_KEY Key;
    while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_NOT_READY) {
      // spin
    }
  }

  return EFI_SUCCESS;
}
