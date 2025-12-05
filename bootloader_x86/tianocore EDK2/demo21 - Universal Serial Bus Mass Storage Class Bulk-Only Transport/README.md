This is a long way from what I wanted to do. But I haven't done anything lately, so it's better than nothing, to get back to learning. USB in general interests me a lot. In the previous demo, I was "going in circle" because I didn't have enough knowledge to delve into the topic enough to understand what it was really about. 
What about the protocols like https://uefi.org/specs/UEFI/2.10/17_Protocols_USB_Support.html#efi-usb-io-protocol-usbgetdevicedescriptor
So I kept trying to enumerate ports on USB Host Controllers and I couldn't reach the endpoints.
<br /><br />
Then I start looking for a posts and better specifications like those:

<h3>More knowledge about USB</h3>
<br />
Beautiful images from a USB packet analyzer and a great post that takes you to a low-level overview of what packets look like and what a handshake looks like. 
These colorful images with blocks of subsequent packets tell a lot about the communication method and what certain packets look like, like "setup," etc. This is the first one
https://www.perytech.com/USB-Enumeration.htm
<br /><br />
Nice pictures of IN/OUT transmission ( transactions )
https://bits4device.wordpress.com/2011/10/14/usb-protocol-device-framework/
<br /><br />
This is a more complex description and more details about the USB topology.
https://www.beyondlogic.org/usbnutshell/usb4.shtml
<br /><br />
Nice and clear pictures and tables with requests structure
https://www.beyondlogic.org/usbnutshell/usb5.shtml
<br /><br />
Universal Serial Bus
Specification Revision 2.0
April 27, 2000 - <b> Here'a a document that I use for this demo. For example page 448 - 11.24.2 Class-specific Requests </b> - I used this to read values ​​for queries etc.
https://wcours.gel.ulaval.ca/GIF1001/old/h20/docs/USB_20.pdf
<br /><br />
XHCI description. Ok, this is new standard than USB 2.0 but nice document to read.
https://www.intel.com/content/dam/www/public/us/en/documents/technical-specifications/extensible-host-controler-interface-usb-xhci.pdf
<br /><br />
There is also information about it on the official Microsoft website
https://learn.microsoft.com/pl-pl/windows-hardware/drivers/usbcon/usb-device-descriptors
<br /><br />
PCI Bus Power Management
Interface Specification
Revision 1.2
March 3, 2004
https://lekensteyn.nl/files/docs/PCI_Power_Management_12.pdf
<br /><br />
<b>THIS IS THE DOCUMENT I USED IN DEMO - IT IS ABOUT A MASS STORAGE DEVICE</b>
https://www.usb.org/sites/default/files/usbmassbulk_10.pdf 
<br /><br />
open source USB Device Viewer
https://github.com/microsoft/Windows-driver-samples/tree/main/usb/usbview
<br /><br />
There's more, these are just a few places where I started looking for technical knowledge on how it works.

<h3>Demo</h3>
<br />
Here is a demo that gives an overview of what it's all about, for example MASS STORAGE class devices. Device from previous demos ( usb pendrive ) VID 0x058F PID 0x6387. But this is very basic demo. 
<br /><br />
The example mass-storage test only sends READ(10) for LBA 0 (one block) via the Bulk-Only Transport CBW/CSW sequence. This is read-only and should not change device contents.
<br /><br />
The test uses UsbBulkTransfer (UEFI API), which is the safest path. It prints the exact CBW, the returned data block (first BlockSize bytes), and the CSW.
<br /><br />
NOT MUCH, but it's better than nothing

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo21%20-%20Universal%20Serial%20Bus%20Mass%20Storage%20Class%20Bulk-Only%20Transport/demo1%20-%20usb%20mass%20storage%20-%2004-12-2025.png?raw=true)

<br /><br />

```
loadimg helloworld.efi
Found '\EFI\Boot\myApps\helloworld.efi'. Run image (YES/NO)?
yes
loadimg: starting image ...

UsbGetDeviceAndInterfaceDesc - enumerate EFI_USB_IO_PROTOCOL handles

[INFO] Found 1 EFI_USB_IO handles

--- USB IO Handle 0 at 7D887040 ---
  Handle: 7D888398
 Device Descriptor:
  Length            : 18
  DescriptorType    : 0x01
  BcdUSB (hex)      : 0x0200
  BcdUSB (bytes)    : 0x02 0x00
  DeviceClass       : 0x00
  DeviceSubClass    : 0x00
  DeviceProtocol    : 0x00
  MaxPacketSize0    : 64
  IdVendor          : 0x058F
  IdProduct         : 0x6387
  BcdDevice (hex)   : 0x0103
  BcdDevice (bytes) : 0x01 0x03
  StrManufacturer   : 1
  StrProduct        : 2
  StrSerialNumber   : 3
  NumConfigurations : 1


[API] Calling UsbGetInterfaceDescriptor() ...
[API] Interface descriptor (active):
  bLength: 9
  bDescriptorType: 0x04
  bInterfaceNumber: 0
  bAlternateSetting: 0
  bNumEndpoints: 2
  bInterfaceClass: 0x08
  bInterfaceSubClass: 0x06
  bInterfaceProtocol: 0x50
  iInterface: 0
[API] UsbGetEndpointDescriptor index=0 ...
  ===>Endpoint Descriptor<===
  bLength: 0x07
  bDescriptorType: 0x05
  bEndpointAddress: 0x01 -> Direction: OUT - EndpointID: 1
  bmAttributes: 0x02 -> Bulk Transfer Type (value=0x02)
  wMaxPacketSize: 0x0200 = 512 bytes
  bInterval: 0x00
[API] UsbGetEndpointDescriptor index=1 ...
  ===>Endpoint Descriptor<===
  bLength: 0x07
  bDescriptorType: 0x05
  bEndpointAddress: 0x82 -> Direction: IN - EndpointID: 2
  bmAttributes: 0x02 -> Bulk Transfer Type (value=0x02)
  wMaxPacketSize: 0x0200 = 512 bytes
  bInterval: 0x00

[INFO] This device matches VID_058F PID_6387 ??? will attempt safe read-only mass-storage test.

[MS TEST] Starting mass storage read-only test for VID=0x058F PID=0x6387 on OUT=0x01 IN=0x82
[MS TEST] CBW (31 bytes) (31 bytes):
0000: 55 53 42 43 01 02 03 04 00 02 00 00 80 00 0A 28
0010: 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
[MS TEST] UsbBulkTransfer OUT endpoint 0x01 send CBW...
[MS TEST] UsbBulkTransfer OUT returned Success  TransferResult=0x00000000  Sent=31 bytes
[MS TEST] UsbBulkTransfer IN endpoint 0x82 read DATA (request 512 bytes)...
[MS TEST] UsbBulkTransfer IN returned Success  TransferResult=0x00000000  Received=512 bytes
[MS TEST] Data block (first 512 bytes or actual) (512 bytes):
0000: 33 C0 8E D0 BC 00 7C 8E C0 8E D8 BE 00 7C BF 00
0010: 06 B9 00 02 FC F3 A4 50 68 1C 06 CB FB B9 04 00
0020: BD BE 07 80 7E 00 00 7C 0B 0F 85 0E 01 83 C5 10
0030: E2 F1 CD 18 88 56 00 55 C6 46 11 05 C6 46 10 00
0040: B4 41 BB AA 55 CD 13 5D 72 0F 81 FB 55 AA 75 09
0050: F7 C1 01 00 74 03 FE 46 10 66 60 80 7E 10 00 74
0060: 26 66 68 00 00 00 00 66 FF 76 08 68 00 00 68 00
0070: 7C 68 01 00 68 10 00 B4 42 8A 56 00 8B F4 CD 13
0080: 9F 83 C4 10 9E EB 14 B8 01 02 BB 00 7C 8A 56 00
0090: 8A 76 01 8A 4E 02 8A 6E 03 CD 13 66 61 73 1C FE
00A0: 4E 11 75 0C 80 7E 00 80 0F 84 8A 00 B2 80 EB 84
00B0: 55 32 E4 8A 56 00 CD 13 5D EB 9E 81 3E FE 7D 55
00C0: AA 75 6E FF 76 00 E8 8D 00 75 17 FA B0 D1 E6 64
00D0: E8 83 00 B0 DF E6 60 E8 7C 00 B0 FF E6 64 E8 75
00E0: 00 FB B8 00 BB CD 1A 66 23 C0 75 3B 66 81 FB 54
00F0: 43 50 41 75 32 81 F9 02 01 72 2C 66 68 07 BB 00
0100: 00 66 68 00 02 00 00 66 68 08 00 00 00 66 53 66
0110: 53 66 55 66 68 00 00 00 00 66 68 00 7C 00 00 66
0120: 61 68 00 00 07 CD 1A 5A 32 F6 EA 00 7C 00 00 CD
0130: 18 A0 B7 07 EB 08 A0 B6 07 EB 03 A0 B5 07 32 E4
0140: 05 00 07 8B F0 AC 3C 00 74 09 BB 07 00 B4 0E CD
0150: 10 EB F2 F4 EB FD 2B C9 E4 64 EB 00 24 02 E0 F8
0160: 24 02 C3 49 6E 76 61 6C 69 64 20 70 61 72 74 69
0170: 74 69 6F 6E 20 74 61 62 6C 65 00 45 72 72 6F 72
0180: 20 6C 6F 61 64 69 6E 67 20 6F 70 65 72 61 74 69
0190: 6E 67 20 73 79 73 74 65 6D 00 4D 69 73 73 69 6E
01A0: 67 20 6F 70 65 72 61 74 69 6E 67 20 73 79 73 74
01B0: 65 6D 00 00 00 63 7B 9A 3B 62 1D 37 00 00 80 82
01C0: 03 00 0B FE 7F FB 00 20 00 00 00 78 7C 00 00 00
01D0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
01E0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
01F0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 55 AA
[MS TEST] UsbBulkTransfer IN endpoint 0x82 read CSW (13 bytes)...
[MS TEST] UsbBulkTransfer CSW returned Success  TransferResult=0x00000000  Received=13 bytes
[MS TEST] CSW (13 bytes):
0000: 55 53 42 53 01 02 03 04 00 00 00 00 00
[MS TEST] CSW Signature=0x53425355 Tag=0x04030201 Residue=0 Status=0x00
[MS TEST] Read-only mass storage test completed (no write performed).
[MS TEST] Mass storage test completed OK.

Done.
loadimg: image exited with status: Success
loadimg: UnloadImage returned: Invalid Parameter
\>
```

Dump from USB Device Viewer

```
[Port2]  :  USB Mass Storage Device


Is Port User Connectable:         yes
Is Port Debug Capable:            no
Companion Port Number:            0
Companion Hub Symbolic Link Name: 
Protocols Supported:
 USB 1.1:                         yes
 USB 2.0:                         yes
 USB 3.0:                         no

Device Power State:               PowerDeviceD0

       ---===>Device Information<===---
English product name: "Mass Storage"

ConnectionStatus:                  
Current Config Value:              0x01  -> Device Bus Speed: High (is not SuperSpeed or higher capable)
Device Address:                    0x02
Open Pipes:                           2

          ===>Device Descriptor<===
bLength:                           0x12
bDescriptorType:                   0x01
bcdUSB:                          0x0200
bDeviceClass:                      0x00  -> This is an Interface Class Defined Device
bDeviceSubClass:                   0x00
bDeviceProtocol:                   0x00
bMaxPacketSize0:                   0x40 = (64) Bytes
idVendor:                        0x058F = Alcor Micro, Corp.
idProduct:                       0x6387
bcdDevice:                       0x0103
iManufacturer:                     0x01
     English (United States)  "Generic"
iProduct:                          0x02
     English (United States)  "Mass Storage"
iSerialNumber:                     0x03
     English (United States)  "EE5B4CC2"
bNumConfigurations:                0x01

          ---===>Open Pipes<===---

          ===>Endpoint Descriptor<===
bLength:                           0x07
bDescriptorType:                   0x05
bEndpointAddress:                  0x01  -> Direction: OUT - EndpointID: 1
bmAttributes:                      0x02  -> Bulk Transfer Type
wMaxPacketSize:                  0x0200 = 0x200 max bytes
bInterval:                         0x00

          ===>Endpoint Descriptor<===
bLength:                           0x07
bDescriptorType:                   0x05
bEndpointAddress:                  0x82  -> Direction: IN - EndpointID: 2
bmAttributes:                      0x02  -> Bulk Transfer Type
wMaxPacketSize:                  0x0200 = 0x200 max bytes
bInterval:                         0x00

       ---===>Full Configuration Descriptor<===---

          ===>Configuration Descriptor<===
bLength:                           0x09
bDescriptorType:                   0x02
wTotalLength:                    0x0020  -> Validated
bNumInterfaces:                    0x01
bConfigurationValue:               0x01
iConfiguration:                    0x00
bmAttributes:                      0x80  -> Bus Powered
MaxPower:                          0x32 = 100 mA

          ===>Interface Descriptor<===
bLength:                           0x09
bDescriptorType:                   0x04
bInterfaceNumber:                  0x00
bAlternateSetting:                 0x00
bNumEndpoints:                     0x02
bInterfaceClass:                   0x08  -> This is a Mass Storage USB Device Interface Class
bInterfaceSubClass:                0x06
bInterfaceProtocol:                0x50
iInterface:                        0x00

          ===>Endpoint Descriptor<===
bLength:                           0x07
bDescriptorType:                   0x05
bEndpointAddress:                  0x01  -> Direction: OUT - EndpointID: 1
bmAttributes:                      0x02  -> Bulk Transfer Type
wMaxPacketSize:                  0x0200 = 0x200 max bytes
bInterval:                         0x00

          ===>Endpoint Descriptor<===
bLength:                           0x07
bDescriptorType:                   0x05
bEndpointAddress:                  0x82  -> Direction: IN - EndpointID: 2
bmAttributes:                      0x02  -> Bulk Transfer Type
wMaxPacketSize:                  0x0200 = 0x200 max bytes
bInterval:                         0x00
```

Code - helloworld.c

```
/** UsbGetDeviceAndInterfaceDesc_WithEndpointsAndMassStorageTest.c
  Extended demo:
   - enumerates EFI_USB_IO handles
   - prints EFI_USB_DEVICE_DESCRIPTOR
   - prints configuration & interface parsing (raw)
   - uses UsbGetInterfaceDescriptor + UsbGetEndpointDescriptor to print EFI_USB_ENDPOINT_DESCRIPTOR fields
   - if device is VID=0x058F PID=0x6387, runs a safe Read-only Mass Storage test
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/PrintLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>

#include <Protocol/UsbIo.h>
#include <IndustryStandard/Usb.h>

typedef struct {
  UINT8  Length;
  UINT8  DescriptorType;
  UINT8  InterfaceNumber;
  UINT8  AlternateSetting;
  UINT8  NumEndpoints;
  UINT8  InterfaceClass;
  UINT8  InterfaceSubClass;
  UINT8  InterfaceProtocol;
  UINT8  Interface; // iInterface
} EFI_USB_INTERFACE_DESCRIPTOR_LOCAL;

STATIC VOID
HexDumpLine (
  IN CHAR16 *Label,
  IN VOID   *Buf,
  IN UINTN  Len
  )
{
  UINT8 *b = (UINT8*)Buf;
  Print (L"%s (%u bytes):\n", Label, (UINT32)Len);
  for (UINTN i = 0; i < Len; ++i) {
    if ((i & 0x0F) == 0) Print (L"%04x: ", (UINT32)i);
    Print (L"%02x ", b[i]);
    if ((i & 0x0F) == 0x0F) Print (L"\n");
  }
  if ((Len & 0x0F) != 0) Print (L"\n");
}

// Print endpoint descriptor fields in human readable form
STATIC VOID
PrintEndpointDescriptor(
  IN EFI_USB_ENDPOINT_DESCRIPTOR *EpDesc
  )
{
  if (EpDesc == NULL) return;

  UINT8 epAddr = EpDesc->EndpointAddress;
  UINT8 dir = (epAddr & 0x80) ? 1 : 0; // 1 = IN, 0 = OUT
  UINT8 epNum = epAddr & 0x0F;

  UINT8 attr = EpDesc->Attributes;
  UINT8 transferType = attr & 0x03;
  CHAR16 *typeStr = L"Unknown";
  switch (transferType) {
    case 0: typeStr = L"Control"; break;
    case 1: typeStr = L"Isochronous"; break;
    case 2: typeStr = L"Bulk"; break;
    case 3: typeStr = L"Interrupt"; break;
  }

  Print (L"  ===>Endpoint Descriptor<===\n");
  Print (L"  bLength: 0x%02x\n", EpDesc->Length);
  Print (L"  bDescriptorType: 0x%02x\n", EpDesc->DescriptorType);
  Print (L"  bEndpointAddress: 0x%02x -> Direction: %s - EndpointID: %u\n", EpDesc->EndpointAddress,
         dir ? L"IN" : L"OUT", epNum);
  Print (L"  bmAttributes: 0x%02x -> %s Transfer Type (value=0x%02x)\n", EpDesc->Attributes, typeStr, transferType);
  Print (L"  wMaxPacketSize: 0x%04x = %u bytes\n", EpDesc->MaxPacketSize, EpDesc->MaxPacketSize);
  Print (L"  bInterval: 0x%02x\n", EpDesc->Interval);
}

// Safe read-only mass storage test using UEFI UsbBulkTransfer (CBW -> Data IN -> CSW)
STATIC EFI_STATUS
TestMassStoragePipes(
  IN EFI_USB_IO_PROTOCOL *UsbIo,
  IN UINT16 IdVendor,
  IN UINT16 IdProduct,
  IN UINT8  BulkOutEp,  // e.g. 0x01
  IN UINT8  BulkInEp,   // e.g. 0x82
  IN UINT32 BlockSize   // typically 512
  )
{
  EFI_STATUS Status;
  UINT32 TransferResult;
  UINTN DataLen;
  Print(L"\n[MS TEST] Starting mass storage read-only test for VID=0x%04x PID=0x%04x on OUT=0x%02x IN=0x%02x\n",
        IdVendor, IdProduct, BulkOutEp, BulkInEp);

  // Build CBW (31 bytes)
  UINT8 cbw[31];
  ZeroMem(cbw, sizeof(cbw));
  cbw[0] = 'U'; cbw[1] = 'S'; cbw[2] = 'B'; cbw[3] = 'C'; // Signature "USBC"
  // tag (arbitrary)
  cbw[4] = 0x01; cbw[5] = 0x02; cbw[6] = 0x03; cbw[7] = 0x04;
  // dCBWDataTransferLength (little endian) -> BlockSize
  cbw[8]  = (UINT8)(BlockSize & 0xFF);
  cbw[9]  = (UINT8)((BlockSize >> 8) & 0xFF);
  cbw[10] = (UINT8)((BlockSize >> 16) & 0xFF);
  cbw[11] = (UINT8)((BlockSize >> 24) & 0xFF);
  cbw[12] = 0x80; // flags: bit7=1 => IN (device->host)
  cbw[13] = 0x00; // bLUN
  cbw[14] = 0x0A; // bCBWLength = 10
  // CDB READ(10) (10 bytes) at cbw[15..24]
  UINT8 cdb[16]; ZeroMem(cdb, sizeof(cdb));
  cdb[0] = 0x28; // READ(10)
  // LBA = 0 (bytes 2..5)
  cdb[2] = 0x00; cdb[3] = 0x00; cdb[4] = 0x00; cdb[5] = 0x00;
  // Transfer length (blocks) = 1 -> bytes 7..8
  cdb[7] = 0x00; cdb[8] = 0x01;
  CopyMem(&cbw[15], cdb, 10);

  // Print CBW
  HexDumpLine(L"[MS TEST] CBW (31 bytes)", cbw, sizeof(cbw));

  // Send CBW via Bulk OUT
  DataLen = sizeof(cbw);
  TransferResult = 0;
  Print(L"[MS TEST] UsbBulkTransfer OUT endpoint 0x%02x send CBW...\n", BulkOutEp);
  Status = UsbIo->UsbBulkTransfer(UsbIo, BulkOutEp, cbw, &DataLen, 5000, &TransferResult);
  Print(L"[MS TEST] UsbBulkTransfer OUT returned %r  TransferResult=0x%08x  Sent=%u bytes\n", Status, TransferResult, DataLen);
  if (EFI_ERROR(Status)) {
    Print(L"[MS TEST] CBW OUT failed, aborting test.\n");
    return Status;
  }

  // Read Data IN (BlockSize)
  UINT8 *dataBuf = AllocateZeroPool(BlockSize);
  if (dataBuf == NULL) {
    Print(L"[MS TEST] Failed to allocate data buffer (%u bytes)\n", BlockSize);
    return EFI_OUT_OF_RESOURCES;
  }
  DataLen = BlockSize;
  TransferResult = 0;
  Print(L"[MS TEST] UsbBulkTransfer IN endpoint 0x%02x read DATA (request %u bytes)...\n", BulkInEp, DataLen);
  Status = UsbIo->UsbBulkTransfer(UsbIo, BulkInEp, dataBuf, &DataLen, 5000, &TransferResult);
  Print(L"[MS TEST] UsbBulkTransfer IN returned %r  TransferResult=0x%08x  Received=%u bytes\n", Status, TransferResult, DataLen);
  if (!EFI_ERROR(Status) && DataLen > 0) {
    HexDumpLine(L"[MS TEST] Data block (first 512 bytes or actual)", dataBuf, DataLen);
  } else {
    Print(L"[MS TEST] Data IN returned no data or error.\n");
  }

  // Read CSW (13 bytes)
  UINT8 csw[13];
  ZeroMem(csw, sizeof(csw));
  DataLen = sizeof(csw);
  TransferResult = 0;
  Print(L"[MS TEST] UsbBulkTransfer IN endpoint 0x%02x read CSW (13 bytes)...\n", BulkInEp);
  Status = UsbIo->UsbBulkTransfer(UsbIo, BulkInEp, csw, &DataLen, 5000, &TransferResult);
  Print(L"[MS TEST] UsbBulkTransfer CSW returned %r  TransferResult=0x%08x  Received=%u bytes\n", Status, TransferResult, DataLen);
  if (!EFI_ERROR(Status) && DataLen > 0) {
    HexDumpLine(L"[MS TEST] CSW", csw, DataLen);
    // Optionally parse CSW: signature/dTag/residue/status
    UINT32 sig = (UINT32)csw[0] | ((UINT32)csw[1] << 8) | ((UINT32)csw[2] << 16) | ((UINT32)csw[3] << 24);
    UINT32 tag = (UINT32)csw[4] | ((UINT32)csw[5] << 8) | ((UINT32)csw[6] << 16) | ((UINT32)csw[7] << 24);
    UINT32 residue = (UINT32)csw[8] | ((UINT32)csw[9] << 8) | ((UINT32)csw[10] << 16) | ((UINT32)csw[11] << 24);
    UINT8 statusByte = csw[12];
    Print(L"[MS TEST] CSW Signature=0x%08x Tag=0x%08x Residue=%u Status=0x%02x\n", sig, tag, residue, statusByte);
  } else {
    Print(L"[MS TEST] Failed to read CSW or no data\n");
  }

  if (dataBuf) FreePool(dataBuf);

  Print(L"[MS TEST] Read-only mass storage test completed (no write performed).\n");
  return EFI_SUCCESS;
}

// Iterates interfaces via UsbGetInterfaceDescriptor and endpoints via UsbGetEndpointDescriptor
STATIC VOID
IterateInterfacesAndEndpoints(
  IN EFI_USB_IO_PROTOCOL *UsbIo,
  IN EFI_USB_DEVICE_DESCRIPTOR *DevDesc
  )
{
  if (UsbIo == NULL) return;

  EFI_STATUS Status;
  EFI_USB_INTERFACE_DESCRIPTOR InterfaceDesc;
  ZeroMem(&InterfaceDesc, sizeof(InterfaceDesc));

  Print(L"\n[API] Calling UsbGetInterfaceDescriptor() ...\n");
  // The protocol provides the interface descriptor of the current active interface
  Status = UsbIo->UsbGetInterfaceDescriptor(UsbIo, &InterfaceDesc);
  if (EFI_ERROR(Status)) {
    Print(L"[API] UsbGetInterfaceDescriptor failed: %r\n", Status);
    return;
  }

  Print(L"[API] Interface descriptor (active):\n");
  Print(L"  bLength: %u\n", InterfaceDesc.Length);
  Print(L"  bDescriptorType: 0x%02x\n", InterfaceDesc.DescriptorType);
  Print(L"  bInterfaceNumber: %u\n", InterfaceDesc.InterfaceNumber);
  Print(L"  bAlternateSetting: %u\n", InterfaceDesc.AlternateSetting);
  Print(L"  bNumEndpoints: %u\n", InterfaceDesc.NumEndpoints);
  Print(L"  bInterfaceClass: 0x%02x\n", InterfaceDesc.InterfaceClass);
  Print(L"  bInterfaceSubClass: 0x%02x\n", InterfaceDesc.InterfaceSubClass);
  Print(L"  bInterfaceProtocol: 0x%02x\n", InterfaceDesc.InterfaceProtocol);
  Print(L"  iInterface: %u\n", InterfaceDesc.Interface);

  // For each endpoint index (0..NumEndpoints-1) call UsbGetEndpointDescriptor
  for (UINTN idx = 0; idx < InterfaceDesc.NumEndpoints; ++idx) {
    EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
    ZeroMem(&EpDesc, sizeof(EpDesc));
    Print(L"[API] UsbGetEndpointDescriptor index=%u ...\n", (UINT32)idx);
    Status = UsbIo->UsbGetEndpointDescriptor(UsbIo, (UINT8)idx, &EpDesc);
    if (EFI_ERROR(Status)) {
      Print(L"[API] UsbGetEndpointDescriptor idx=%u failed: %r\n", (UINT32)idx, Status);
      continue;
    }
    PrintEndpointDescriptor(&EpDesc);
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
  EFI_HANDLE *HandleBuffer = NULL;
  UINTN HandleCount = 0;

  Print (L"\nUsbGetDeviceAndInterfaceDesc - enumerate EFI_USB_IO_PROTOCOL handles\n\n");

  Status = gBS->LocateHandleBuffer(
                 ByProtocol,
                 &gEfiUsbIoProtocolGuid,
                 NULL,
                 &HandleCount,
                 &HandleBuffer
                 );
  if (EFI_ERROR (Status)) {
    Print (L"[ERR] LocateHandleBuffer(ByProtocol, EFI_USB_IO_PROTOCOL) failed: %r\n", Status);
    return Status;
  }

  if (HandleCount == 0) {
    Print (L"[INFO] No EFI_USB_IO_PROTOCOL handles found.\n");
    return EFI_SUCCESS;
  }

  Print (L"[INFO] Found %u EFI_USB_IO handles\n\n", (UINT32)HandleCount);

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol (HandleBuffer[i], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
    if (EFI_ERROR (Status) || UsbIo == NULL) {
      Print (L"[WARN] Handle %u: failed to open EFI_USB_IO_PROTOCOL: %r\n", (UINT32)i, Status);
      continue;
    }

    Print (L"--- USB IO Handle %u at %p ---\n", (UINT32)i, UsbIo);
    Print (L"  Handle: %p\n", HandleBuffer[i]);

    // 1) Get Device Descriptor via EFI_USB_IO_PROTOCOL
    EFI_USB_DEVICE_DESCRIPTOR DevDesc;
    ZeroMem (&DevDesc, sizeof(DevDesc));
    Status = UsbIo->UsbGetDeviceDescriptor (UsbIo, &DevDesc);
    if (EFI_ERROR (Status)) {
      Print (L"[ERR] UsbGetDeviceDescriptor failed: %r\n\n", Status);
      continue;
    }

    // Print device descriptor fields (including readable BCD)
    UINT8 bcdUsbHigh = (UINT8)((DevDesc.BcdUSB >> 8) & 0xFF);
    UINT8 bcdUsbLow  = (UINT8)(DevDesc.BcdUSB & 0xFF);
    UINT8 bcdDevHigh = (UINT8)((DevDesc.BcdDevice >> 8) & 0xFF);
    UINT8 bcdDevLow  = (UINT8)(DevDesc.BcdDevice & 0xFF);

    Print (L" Device Descriptor:\n");
    Print (L"  Length            : %u\n", DevDesc.Length);
    Print (L"  DescriptorType    : 0x%02x\n", DevDesc.DescriptorType);
    Print (L"  BcdUSB (hex)      : 0x%04x\n", DevDesc.BcdUSB);
    Print (L"  BcdUSB (bytes)    : 0x%02x 0x%02x\n", bcdUsbHigh, bcdUsbLow);
    Print (L"  DeviceClass       : 0x%02x\n", DevDesc.DeviceClass);
    Print (L"  DeviceSubClass    : 0x%02x\n", DevDesc.DeviceSubClass);
    Print (L"  DeviceProtocol    : 0x%02x\n", DevDesc.DeviceProtocol);
    Print (L"  MaxPacketSize0    : %u\n", DevDesc.MaxPacketSize0);
    Print (L"  IdVendor          : 0x%04x\n", DevDesc.IdVendor);
    Print (L"  IdProduct         : 0x%04x\n", DevDesc.IdProduct);
    Print (L"  BcdDevice (hex)   : 0x%04x\n", DevDesc.BcdDevice);
    Print (L"  BcdDevice (bytes) : 0x%02x 0x%02x\n", bcdDevHigh, bcdDevLow);
    Print (L"  StrManufacturer   : %u\n", DevDesc.StrManufacturer);
    Print (L"  StrProduct        : %u\n", DevDesc.StrProduct);
    Print (L"  StrSerialNumber   : %u\n", DevDesc.StrSerialNumber);
    Print (L"  NumConfigurations : %u\n\n", DevDesc.NumConfigurations);

    //
    // 2) Use the protocol helpers to get interface & endpoint descriptors
    //
    IterateInterfacesAndEndpoints(UsbIo, &DevDesc);

    //
    // 3) If this is the mass-storage device you specified, perform the safe read-only test
    //
    if (DevDesc.IdVendor == 0x058F && DevDesc.IdProduct == 0x6387) {
      Print(L"\n[INFO] This device matches VID_058F PID_6387 — will attempt safe read-only mass-storage test.\n");
      // Per your example, endpoints are:
      // OUT 0x01 (bulk out), IN 0x82 (bulk in), wMaxPacketSize=0x0200 (512)
      UINT8 epOut = 0x01;
      UINT8 epIn  = 0x82;
      UINT32 blockSize = 512;

      // Call test function (read-only)
      EFI_STATUS msStatus = TestMassStoragePipes(UsbIo, DevDesc.IdVendor, DevDesc.IdProduct, epOut, epIn, blockSize);
      if (EFI_ERROR(msStatus)) {
        Print(L"[MS TEST] Mass storage test returned error: %r\n", msStatus);
      } else {
        Print(L"[MS TEST] Mass storage test completed OK.\n");
      }
    } else {
      Print(L"[INFO] Device VID/PID does not match the specified mass-storage example, skipping MS test.\n");
    }

    Print (L"\n"); // spacer
  } // for handles

  if (HandleBuffer) FreePool (HandleBuffer);

  Print (L"Done.\n");
  return EFI_SUCCESS;
}
```

INF

```
## @file
#  Sample UEFI Application Reference EDKII Module.
#
#  This is a sample shell application that will print "UEFI Hello World!" to the
#  UEFI Console based on PCD setting.
#
#  It demos how to use EDKII PCD mechanism to make code more flexible.
#
#  Copyright (c) 2008 - 2018, Intel Corporation. All rights reserved.<BR>
#
#  SPDX-License-Identifier: BSD-2-Clause-Patent
#
#
##

[Defines]
  INF_VERSION                    = 0x00010005
  BASE_NAME                      = HelloWorld
  MODULE_UNI_FILE                = HelloWorld.uni
  FILE_GUID                      = 6987936E-ED34-44db-AE97-1FA5E4ED2116
  MODULE_TYPE                    = UEFI_APPLICATION
  VERSION_STRING                 = 1.0
  ENTRY_POINT                    = UefiMain

#
#  This flag specifies whether HII resource section is generated into PE image.
#
  UEFI_HII_RESOURCE_SECTION      = TRUE

#
# The following information is for reference only and not required by the build tools.
#
#  VALID_ARCHITECTURES           = IA32 X64 EBC
#

[Sources]
  HelloWorld.c
  HelloWorldStr.uni

[Packages]
  MdePkg/MdePkg.dec
  MdeModulePkg/MdeModulePkg.dec

[LibraryClasses]
  UefiApplicationEntryPoint
  PcdLib
  UefiBootServicesTableLib
  UefiLib
  PrintLib
  MemoryAllocationLib
  BaseLib
  BaseMemoryLib

[FeaturePcd]
  gEfiMdeModulePkgTokenSpaceGuid.PcdHelloWorldPrintEnable   ## CONSUMES

[Pcd]
  gEfiMdeModulePkgTokenSpaceGuid.PcdHelloWorldPrintString   ## SOMETIMES_CONSUMES
  gEfiMdeModulePkgTokenSpaceGuid.PcdHelloWorldPrintTimes    ## SOMETIMES_CONSUMES

[UserExtensions.TianoCore."ExtraFiles"]
  HelloWorldExtra.uni

[Protocols]
  gEfiPciIoProtocolGuid        ## CONSUMES
  gEfiUsb2HcProtocolGuid       ## CONSUMES
  gEfiUsbIoProtocolGuid
  gEfiUsbFunctionIoProtocolGuid      ## CONSUMES
  gEfiDevicePathToTextProtocolGuid  
  gEfiDevicePathProtocolGuid  
  gEfiSmbusHcProtocolGuid
  gEfiDriverBindingProtocolGuid
  gEfiLoadedImageProtocolGuid
  gEfiBlockIoProtocolGuid
  
```


