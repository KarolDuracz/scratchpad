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
#include <Guid/GlobalVariable.h>
#include <IndustryStandard/Usb.h>

#define TIMEOUT_MS 5000

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
      EFI_USB_CONFIG_DESCRIPTOR ConfigDesc;
      Status = UsbIo->UsbGetConfigDescriptor (UsbIo, &ConfigDesc);
      PrintStatus (L"UsbGetConfigDescriptor", Status);
      if (!EFI_ERROR (Status)) {
        Print (L"  wTotalLength=%u bNumInterfaces=%u bConfigurationValue=%u\n",
               ConfigDesc.TotalLength, ConfigDesc.NumInterfaces, ConfigDesc.ConfigurationValue);
        // Dump first 64 bytes for inspection (if present)
        UINTN DumpLen = (ConfigDesc.TotalLength > 64) ? 64 : ConfigDesc.TotalLength;
        //DumpHex (L"  ConfigDescriptor(raw)", (UINT8*)ConfigDesc, DumpLen);
		Print(L" skipping DumpHex %u \n", DumpLen);
        //FreePool (ConfigDesc);
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
      PrintStatus (L"UsbPortReset (note: this will re-enumerate device in some implementations)", Status);
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

  // Stage 3: find and exercise EFI_USB_IO_PROTOCOL devices
  CheckUsbDevices ();

  Print (L"UsbDiagApp: Completed checks. You can re-run to capture new events.\n");

	// wait for key
  Print(L"Done. Press any key to exit...\n");
  EFI_INPUT_KEY Key;
  while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) != EFI_SUCCESS) {
    // spin
  }


  return EFI_SUCCESS;
}
