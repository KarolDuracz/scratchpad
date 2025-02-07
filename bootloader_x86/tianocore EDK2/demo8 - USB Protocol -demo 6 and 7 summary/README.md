<h2>What is difference between EFI_USB_IO_PROTOCOL and EFI_USB2_HC_ PROTOCOL</h2>
From ChatGPT: <br /> <br />
Both EFI_USB_IO_PROTOCOL and EFI_USB2_HC_PROTOCOL are UEFI protocols for handling USB, but they serve different purposes and operate at different levels.

1️EFI_USB_IO_PROTOCOL (High-Level USB Access)
What it does:

Provides high-level functions for USB device communication.
Used after a USB device has been detected and assigned an address.
Allows you to interact directly with USB devices (e.g., reading descriptors, sending commands).
Use cases:

Retrieving USB device descriptors.
Reading USB configurations.
Sending and receiving data to/from USB devices (e.g., keyboards, mice, flash drives).
Example functions:

```
EFI_STATUS (EFIAPI *UsbGetDeviceDescriptor)(
  IN EFI_USB_IO_PROTOCOL *This,
  OUT EFI_USB_DEVICE_DESCRIPTOR *DeviceDescriptor
);
```

Retrieves the device descriptor without requiring low-level control transfers.
<hr>
EFI_USB2_HC_PROTOCOL (Low-Level Host Controller Access)
What it does:

Provides low-level access to the USB Host Controller (HC).
Used for enumerating USB devices, managing ports, and sending raw USB transactions.
Allows direct USB control transfers to communicate with devices.
Use cases:

Detecting newly connected USB devices.
Resetting USB ports and initializing devices.
Performing raw control, bulk, and interrupt transfers.
Example functions:

```
EFI_STATUS (EFIAPI *ControlTransfer)(
  IN EFI_USB2_HC_PROTOCOL *This,
  IN UINT8 DeviceAddress,
  IN UINT8 DeviceSpeed,
  IN UINTN MaximumPacketLength,
  IN EFI_USB_DEVICE_REQUEST *Request,
  IN EFI_USB_DATA_DIRECTION TransferDirection,
  IN OUT VOID *Data OPTIONAL,
  IN OUT UINTN *DataLength OPTIONAL,
  IN UINTN TimeOut,
  IN EFI_USB2_HC_TRANSACTION_TRANSLATOR *Translator,
  OUT UINT32 *TransferResult
);
```

Sends low-level control transfers to the USB device.
<br /><br />
![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo8%20-%20USB%20Protocol%20-demo%206%20and%207%20summary/summary%20demo8%20gpt.png?raw=true)

<hr>

<h2>Fixes for demo 6 and 7</h2>
https://uefi.org/specs/UEFI/2.10/17_Protocols_USB_Support.html#efi-usb-io-protocol-usbgetdevicedescriptor
Lesson from it - There are 2 different protocols. And based on the description of both protocols in the next demo I can look for confirmation whether it is actually as it "says".<br /><br />
Line 406 - https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo8%20-%20USB%20Protocol%20-demo%206%20and%207%20summary/HelloWorld.c#L406 - execution line 599. HelloWorld.efi is for this demo 8, like on these pictures.
<br /><br />
Virtual Box

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo8%20-%20USB%20Protocol%20-demo%206%20and%207%20summary/61%20-%2007-02-2025%20-%20cd.png?raw=true)

Real HW

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/bootloader_x86/tianocore%20EDK2/demo8%20-%20USB%20Protocol%20-demo%206%20and%207%20summary/62%20-%2007-02-2025%20-%20few%20hours%20later.png)
