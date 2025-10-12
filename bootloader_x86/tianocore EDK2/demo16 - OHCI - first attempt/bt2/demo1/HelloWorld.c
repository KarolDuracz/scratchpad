/**
  Find USB devices that are children of PCI USB host controllers (PCI -> root hub -> USB device).
  - Enumerates PCI devices (via EFI_PCI_IO_PROTOCOL).
  - Selects PCI devices with BaseClass=0x0C (Serial Bus) and SubClass=0x03 (USB).
  - For each such PCI host, grabs its DevicePath and finds all EFI_USB_IO_PROTOCOL handles
    whose DevicePath is a binary child of that PCI DevicePath (prefix match).
  - For each matched USB device: prints VendorID/ProductID and enumerates interface descriptors
    from the active configuration (UsbGetConfigDescriptor).
  - This shows per-interface numbers (these map to Windows per-interface PnP IDs like %0%N).
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
#include <Protocol/UsbIo.h>
#include <IndustryStandard/Usb.h>

#ifndef END_DEVICE_PATH_TYPE
#define END_DEVICE_PATH_TYPE 0x7F
#endif
#ifndef END_ENTIRE_DEVICE_PATH_SUBTYPE
#define END_ENTIRE_DEVICE_PATH_SUBTYPE 0xFF
#endif

// Compute device-path total length (walk nodes until End node)
STATIC
UINTN
GetDevicePathSizeBytes (
  IN EFI_DEVICE_PATH_PROTOCOL *DevicePath
  )
{
  if (DevicePath == NULL) {
    return 0;
  }
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

// Binary prefix compare
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

// Print interface descriptor contents (from raw descriptor buffer)
STATIC
VOID
PrintUsbInterfacesFromConfigBlob (
  IN UINT8 *Buf,
  IN UINTN  Len
  )
{
  UINTN idx = 0;
  while (idx + 2 <= Len) {
    UINT8 bLen  = Buf[idx + 0];
    UINT8 bType = Buf[idx + 1];
    if (bLen < 2) break;
    if (idx + bLen > Len) break;

    if (bType == USB_DESC_TYPE_INTERFACE) {
      if (bLen >= 9) {
        UINT8 bInterfaceNumber   = Buf[idx + 2];
        UINT8 bAlternateSetting  = Buf[idx + 3];
        UINT8 bNumEndpoints      = Buf[idx + 4];
        UINT8 bInterfaceClass    = Buf[idx + 5];
        UINT8 bInterfaceSubClass = Buf[idx + 6];
        UINT8 bInterfaceProtocol = Buf[idx + 7];
        UINT8 iInterface         = Buf[idx + 8];

        Print(
          L"      Interface %u (alt %u): Class=0x%02x Sub=0x%02x Prot=0x%02x Endpoints=%u iInterface=%u\n",
          (UINT32)bInterfaceNumber,
          (UINT32)bAlternateSetting,
          (UINT32)bInterfaceClass,
          (UINT32)bInterfaceSubClass,
          (UINT32)bInterfaceProtocol,
          (UINT32)bNumEndpoints,
          (UINT32)iInterface
        );

        if (bInterfaceClass == 0xE0) {
          Print(L"        -> Wireless Controller (likely Bluetooth). Windows instance: %%0%%%u\n", (UINT32)bInterfaceNumber);
        }
      }
    }

    idx += bLen;
  }
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

  // Find all handles that support EFI_PCI_IO_PROTOCOL
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Print(L"No PCI devices found or error: %r\n", Status);
    return EFI_SUCCESS;
  }

  // Prepare list of USB device handles (EFI_USB_IO_PROTOCOL) for later matching
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
  if (EFI_ERROR(Status)) {
    UsbHandles = NULL;
    UsbCount = 0;
  }

  // DevicePath->Text for nicer output (optional)
  EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpToText = NULL;
  gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID**)&DpToText);

  Print(L"Scanning %u PCI devices for USB host controllers...\n\n", (UINT32)PciCount);

  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) {
      continue;
    }

    // Read 32-bit at offset 0x08 to get ClassCode/Subclass/ProgIF/Revision
    UINT32 ClassReg = 0;
    //Status = PciIo->Pci.Read(PciIo, EfiPciWidthUint32, 0x08, 1, &ClassReg);
	Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassReg);
    if (EFI_ERROR(Status)) {
      continue;
    }

    UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
    UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
    UINT8 ProgIf    = (ClassReg >> 8)  & 0xFF;

    // Only interested in USB host controllers: BaseClass = 0x0C, SubClass = 0x03
    if (BaseClass == 0x0C && SubClass == 0x03) {
      // Print PCI device info
      Print(L"PCI USB Host Controller found (handle=0x%p): Class=0x%02x Sub=0x%02x ProgIf=0x%02x\n",
            PciHandles[i], (UINT32)BaseClass, (UINT32)SubClass, (UINT32)ProgIf);

      // Get its device path
      EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
      Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
      if (!EFI_ERROR(Status) && PciDp != NULL) {
        if (DpToText != NULL) {
          CHAR16 *Text = DpToText->ConvertDevicePathToText(PciDp, FALSE, FALSE);
          if (Text) {
            Print(L"  PCI DevicePath: %s\n", Text);
            FreePool(Text);
          }
        } else {
          Print(L"  PCI DevicePath: (present)\n");
        }
      }

      // Compute size of PCI DP for prefix matching
      UINTN PciDpSize = (PciDp) ? GetDevicePathSizeBytes(PciDp) : 0;

      // Now find USB devices whose DevicePath starts with this PCI device path
      Print(L"  Looking for USB devices under this host (scanning %u USB handles)...\n", (UINT32)UsbCount);
      for (UINTN u = 0; u < UsbCount; ++u) {
        EFI_DEVICE_PATH_PROTOCOL *UsbDp = NULL;
        Status = gBS->HandleProtocol(UsbHandles[u], &gEfiDevicePathProtocolGuid, (VOID **)&UsbDp);
        if (EFI_ERROR(Status) || UsbDp == NULL) {
          continue;
        }

        // Check binary prefix
        UINTN UsbDpSize = GetDevicePathSizeBytes(UsbDp);
        if (PciDp != NULL && UsbDpSize >= PciDpSize && DevicePathIsPrefixBinary(PciDp, PciDpSize, UsbDp, UsbDpSize)) {
          // This USB device is under the PCI host controller
          Print(L"    USB device handle=0x%p (DevicePath is child of PCI host)\n", UsbHandles[u]);

          // Print textual device path if available
          if (DpToText != NULL) {
            CHAR16 *Text = DpToText->ConvertDevicePathToText(UsbDp, FALSE, FALSE);
            if (Text) {
              Print(L"      DevicePath: %s\n", Text);
              FreePool(Text);
            }
          }

          // Query USB device descriptor
          EFI_USB_IO_PROTOCOL *UsbIo = NULL;
          Status = gBS->HandleProtocol(UsbHandles[u], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
          if (EFI_ERROR(Status) || UsbIo == NULL) {
            Print(L"      (failed to get EFI_USB_IO_PROTOCOL: %r)\n", Status);
            continue;
          }

          EFI_USB_DEVICE_DESCRIPTOR DevDesc;
          ZeroMem(&DevDesc, sizeof(DevDesc));
          Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
          if (EFI_ERROR(Status)) {
            Print(L"      UsbGetDeviceDescriptor failed: %r\n", Status);
          } else {
            Print(L"      VendorId=0x%04x ProductId=0x%04x DeviceClass=0x%02x\n",
                  DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
          }

          // Get full active configuration descriptor (UsbGetConfigDescriptor)
          //EFI_USB_CONFIG_DESCRIPTOR *CfgDesc = NULL;
		  EFI_USB_CONFIG_DESCRIPTOR CfgDesc;
          Status = UsbIo->UsbGetConfigDescriptor(UsbIo, &CfgDesc);
          
		  if (!EFI_ERROR(Status)) {
            // Use raw bytes pointer and extract wTotalLength from offset 2..3
            UINT8 *Raw = (UINT8 *)&CfgDesc;
            UINTN totalLen = 0;
            // make sure we can safely read offset 2/3
            if (Raw != NULL) {
              totalLen = (UINTN)((UINT16)Raw[2] | ((UINT16)Raw[3] << 8));
              Print(L"      Active configuration total length = %u bytes\n", (UINT32)totalLen);
              if (totalLen > 0 && totalLen <= 64 * 1024) {
                PrintUsbInterfacesFromConfigBlob(Raw, totalLen);
              } else {
                // fallback: attempt parse from a reasonable buffer length (CfgDesc may be smaller)
                // Try to parse using a safe upper bound: sizeof(*CfgDesc) is small; but we'll parse up to 4096
                UINTN safeLen = 4096;
                Print(L"      Warning: abnormal totalLen=%u; parsing up to %u bytes (best-effort)\n", (UINT32)totalLen, (UINT32)safeLen);
                PrintUsbInterfacesFromConfigBlob(Raw, (totalLen > 0 && totalLen <= safeLen) ? totalLen : safeLen);
              }
            }
            //FreePool(CfgDesc); // ???????? 
            //CfgDesc = NULL;
          } else {
            Print(L"      UsbGetConfigDescriptor failed: %r\n", Status);
            // If UsbGetConfigDescriptor isn't available, a fallback would be to issue a control transfer
            // GET_DESCRIPTOR (Configuration) in two calls (first 9 bytes to get wTotalLength, then full buffer).
            // That fallback is more verbose; add if needed.
          }
        }
      } // for each USB handle
      Print(L"\n");
    } // if is USB host controller
  } // for each PCI handle

  // Free buffers
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
