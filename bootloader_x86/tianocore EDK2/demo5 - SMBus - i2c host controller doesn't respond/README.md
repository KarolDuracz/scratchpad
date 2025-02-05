<h2>SMBus protocol</h2>

Next chunk of uefi code <br /><br />
I tried the I2C protocol EFI_I2C_HOST_PROTOCOL but it doesn't work, it doesn't detect the host controller in line 216. It returns the error that I defined as 0xFFFFFFFF
https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo5%20-%20SMBus%20-%20i2c%20host%20controller%20doesn't%20respond/HelloWorld.c#L216
<br /><br />
So if not I2C then SMBus and the EFI_SMBUS_HC_PROTOCOL protocol which is detected in Windows. In line 238
https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo5%20-%20SMBus%20-%20i2c%20host%20controller%20doesn't%20respond/HelloWorld.c#L238
<br /><br />
Everything looked ok, so I iterated through the addresses. I don't checked if something is between 0x50 - 0x57, I just displayed everything that is in between 0x03 - 0x77 to see what it finds. In this test I was specifically looking for a DISPLAYS devices. And as you can see in the picture there is something between 0x50 - 0x57. It is possible that these are 2 devices for displaying the image, but I have not done a loop to EDID yet. I don't know yet what these two devices are doing at addresses 0x33 and x045.

```
#define SMBUS_MIN_ADDR  0x03  // Lowest SMBus address
#define SMBUS_MAX_ADDR  0x77  // Highest SMBus address
#define EDID_SMBUS_MIN  0x50  // EDID device range
#define EDID_SMBUS_MAX  0x57
```

The code execution is on lines 378 - 380


![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo5%20-%20SMBus%20-%20i2c%20host%20controller%20doesn't%20respond/1738778791687.jpg?raw=true)

EDID from Windows for integrated laptop display. The first byte doesn't match. I don't know what this protocol read here, I need to study the documentation better
, but it's probably some address for another protocol, because the EDID in the revision that my displays have uses the beginning 00,ff,ff,ff,ff,ff,ff,00. I think this sequence is the same for all revisions, but the first byte 0x92 at the beginning doesn't match the EDID. It's something else. That's by the way.

```
Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\DISPLAY\AUO139E\4&302c6972&0&UID67568640\Device Parameters]
"EDID"=hex:00,ff,ff,ff,ff,ff,ff,00,06,af,9e,13,00,00,00,00,01,13,01,03,80,26,\
  15,78,0a,c4,95,9e,57,53,92,26,0f,50,54,00,00,00,01,01,01,01,01,01,01,01,01,\
  01,01,01,01,01,01,01,f8,2a,40,90,61,84,0c,30,30,20,36,00,7e,d6,10,00,00,18,\
  00,00,00,0f,00,00,00,00,00,00,00,00,00,00,00,00,00,20,00,00,00,fe,00,41,55,\
  4f,0a,20,20,20,20,20,20,20,20,20,00,00,00,fe,00,42,31,37,33,52,57,30,31,20,\
  56,33,20,0a,00,26

[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\DISPLAY\AUO139E\4&302c6972&0&UID67568640\Device Parameters\e5b3b5ac-9725-4f78-963f-03dfb1d828c7]
```

Example code to read EDID data

```
EFI_STATUS ReadEdidData(EFI_SMBUS_HC_PROTOCOL *Smbus, UINT8 Address) {
    EFI_SMBUS_DEVICE_ADDRESS SmbusDevice;
    EFI_SMBUS_DEVICE_COMMAND Command = 0x00;  // Command to read EDID data
    UINTN Length = 128;  // EDID data length (128 bytes)
    UINT8 EdidData[128]; // Buffer to store EDID data
    EFI_STATUS Status;

    SmbusDevice.SmbusDeviceAddress = Address;

    // Execute the read command
    Status = Smbus->Execute(Smbus, SmbusDevice, Command, EfiSmbusReadByte, FALSE, &Length, EdidData);
    
    if (EFI_ERROR(Status)) {
        Print(L"Failed to read from address 0x%02X\n", Address);
        return Status;
    }

    // Process the EDID data (for display devices)
    Print(L"EDID Data from device at 0x%02X:\n", Address);
    for (UINTN i = 0; i < Length; i++) {
        Print(L"%02X ", EdidData[i]);
    }
    Print(L"\n");

    return EFI_SUCCESS;
}
```

References:<br />
https://uefi.org/specs/PI/1.8/V5_SMBusPPI_Code_Definitions.html <br />
https://www.intel.fr/content/dam/doc/reference-guide/efi-smbus-host-controller-protocol-specification.pdf<br />
https://smbus.org/specs/ <-- link from intel docs to --->The Execute() function provides a standard way to execute an operation as defined in the 
System Management Bus (SMBus) Specification. The resulting transaction will be either that the 
SMBus slave devices accept this transaction or that this function returns with error.<br />
https://glenwing.github.io/docs/VESA-EEDID-A2.pdf<br />
https://www.extron.com/article/uedid<br />
<hr>
One more thing. SMBus in my laptop configuration is on PCI Dev 31, Fun 3.| PAGE 739 | 
6-chipset-c200-chipset-datasheet.pdf - 2011 - 
https://www.intel.com/content/dam/www/public/us/en/documents/datasheets/6-chipset-c200-chipset-datasheet.pdf
