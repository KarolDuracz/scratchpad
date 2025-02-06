/** @file
  This sample application bases on HelloWorld PCD setting
  to print "UEFI Hello World!" to the UEFI Console.

  Copyright (c) 2006 - 2018, Intel Corporation. All rights reserved.<BR>
  SPDX-License-Identifier: BSD-2-Clause-Patent

**/

#include <Uefi.h>
#include <Library/PcdLib.h>
#include <Library/UefiLib.h>
#include <Library/BaseLib.h>
#include <Library/PrintLib.h>
#include <Library/UefiApplicationEntryPoint.h>

// GOP
#include <Library/DebugLib.h> // added inf
#include <Protocol/GraphicsOutput.h>
#include <Library/UefiBootServicesTableLib.h>
//#include <Library/TimerLib.h> // added inf

// PCI
#include <Protocol/PciRootBridgeIo.h>

// IO
#include <Library/IoLib.h>

// I2C
//#include <Library/UefiBootServicesTableLib.h> // gop also use this
#include <Protocol/I2cHost.h>
#define I2C_MAX_ADDRESS 0x7

// SMBus
#include <Protocol/SmbusHC.h>

// USB
#include <Protocol/Usb2HostController.h>

// START EFI CODE => global variables
extern UINT64 _vartest;
extern UINT64 _vartest2;
extern UINT64 _vartest3;
extern UINT64 _vartest4_timer1;
UINT64 *ptr;

static EFI_GRAPHICS_OUTPUT_PROTOCOL *mGraphicsOuput = NULL;

EFI_GRAPHICS_OUTPUT_BLT_PIXEL white = { 255, 255, 255, 0 };
EFI_GRAPHICS_OUTPUT_BLT_PIXEL blue = { 255, 234, 0, 0 };

//
// String token ID of help message text.
// Shell supports to find help message in the resource section of an application image if
// .MAN file is not found. This global variable is added to make build tool recognizes
// that the help string is consumed by user and then build tool will add the string into
// the resource section. Thus the application can use '-?' option to show help message in
// Shell.
//
GLOBAL_REMOVE_IF_UNREFERENCED EFI_STRING_ID  mStringHelpTokenId = STRING_TOKEN (STR_HELLO_WORLD_HELP_INFORMATION);

/* functions prototypes and declarations - maybe this is needed in headers or external libraries but... */
EFI_STATUS
EFIAPI
_print (
	IN EFI_SYSTEM_TABLE *SystemTable, 	// pointer to calling SystemTable->ConOut->OutputString
	IN UINT32 value,					// value to convert to hex and print on console
	IN UINT64 value64,					// for 64 bits values
	IN UINT32 mode,						// mode 8 - for 32 bits | mode 16 - for 64 bit values - if null default is 8
	IN UINT32 x,						// x cursor position on screen 
	IN UINT32 y,						// y cursor position on screen
	IN UINT32 endTextMode				// 0 - \r\n\0  means new line || 1 - ' ' means space	
	);


EFI_STATUS EFIAPI _print(IN EFI_SYSTEM_TABLE  *SystemTable, IN UINT32 value, IN UINT64 value64, 
		IN UINT32 mode, IN UINT32 x, IN UINT32 y, IN UINT32 endTextMode)
{
	
	// convert value to hex and print on console
	CHAR16 buffer[256]; 	
	CHAR16 *hexchar = L"0123456789ABCDEF";
	
	// 32 bits value
	if (mode == 0) {
		for (INTN j = 0; j < 8; j++) {
			buffer[7 - j] = hexchar[(value >> (j * 4)) & 0xf];
		}
		if(endTextMode == 0) {
			buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
		}
		if (endTextMode == 1) {
			buffer[8] = L' '; buffer[9] = L'\0';
		}
	}
	
	// 64 bits value
	if (mode == 1) {
		for (INTN j = 0; j < 16; j++) {
			buffer[15 - j] = hexchar[(value64 >> (j * 4)) & 0xf];
		}
		if(endTextMode == 0) {
			buffer[16] = L'\r'; buffer[17] = L'\n'; buffer[18] = L'\0'; 
		}
		if(endTextMode == 1) {
			buffer[16] = L' '; buffer[18] = L'\0'; 
		}
	}
	
	// TODO mode, x, y handles
	//if (x > 0 || y > 0) {
	//	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, x, y);
	//}
		
	SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
	
	return EFI_SUCCESS;
}

EFI_STATUS ReadPciConfig(
	EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *PciRootBridgeIo,
	UINT8 Bus,
	UINT8 Device,
	UINT8 Function,
	UINT8 Offset,
	UINT32 *Value
)
{
	EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL_PCI_ADDRESS Address = {
		.Bus = Bus,
		.Device = Device,
		.Function = Function,
		.ExtendedRegister = Offset
	};
	
	return PciRootBridgeIo->Pci.Read(PciRootBridgeIo, EfiPciWidthUint32, *(UINT64 *)&Address, 1, Value);
}

EFI_STATUS ReadPciConfig64(
	EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *PciRootBridgeIo,
	UINT8 Bus,
	UINT8 Device,
	UINT8 Function,
	UINT8 Offset,
	UINT64 *Value
)
{
	EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL_PCI_ADDRESS Address = {
		.Bus = Bus,
		.Device = Device,
		.Function = Function,
		.ExtendedRegister = Offset
	};
	
	return PciRootBridgeIo->Pci.Read(PciRootBridgeIo, EfiPciWidthUint64, *(UINT64 *)&Address, 1, Value);
}

#define VGA_CRTC_ADDRESS_PORT 0x3d4
#define VGA_CRTC_DATA_PORT	0x3d5
#define VGA_ATTRIBUTE_ADDRESS_PORT 0x3c0
#define VGA_ATTRIBUTE_DATA_PORT 0x3c1

// GR10 - Address Mapping
// first need to call to index 0x0 - 
#define VGA_GRX 0x3ce 
#define VGA_CR10_BASE 0x3cf // this is index 10


// IO VGA
UINT8 ReadVgaCrtReg(UINT8 index)
{
	IoWrite8(VGA_CRTC_ADDRESS_PORT, index);
	return IoRead8(VGA_CRTC_DATA_PORT);
}

UINT8 ReadVgaCrtAttr(UINT8 index)
{
	IoWrite8(VGA_ATTRIBUTE_ADDRESS_PORT, index);
	return IoRead8(VGA_ATTRIBUTE_DATA_PORT);
}

// GRX
UINT8 ReadGRX(UINT8 index)
{
	IoWrite8(VGA_GRX, index);
	return IoRead8(VGA_CR10_BASE);
}

// VGA current settings read - FULL DETAIL ABOUT DISPLAY CONFIGURATION
UINT8 ReadVga(UINT8 index)
{
	// 0x3d4 is the base address for this demo function
	IoWrite8(VGA_CRTC_ADDRESS_PORT, index);
	// the same as previous - return addres is 0x3d5 for this case
	return IoRead8(VGA_CRTC_DATA_PORT);
}

/* I2c stuff */
EFI_STATUS FindI2cHostController(IN EFI_SYSTEM_TABLE  *SystemTable, EFI_I2C_HOST_PROTOCOL **I2cHost)
{
	EFI_STATUS Status;
	EFI_HANDLE *HandleBuffer;
	UINTN HandleCount;
	
	// first find SMBus insted of i2c because for me I2c host protocol return 0xFFFFFFFF code
	// so I check this and returned 0x2F... that means, it works.
	// next is to write function to examine SMBus 5-02-2025 
	Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiSmbusHcProtocolGuid, NULL, &HandleCount, &HandleBuffer);
	if (EFI_ERROR(Status)) {
		// no i2c host controller found
		_print(SystemTable, (UINT32)0x1FFFFFFF, 0, 0, 0, 0, 0);
		CpuDeadLoop();
	} else {
		_print(SystemTable, (UINT32)0x2FFFFFFF, 0, 0, 0, 0, 0);
		CpuDeadLoop();
	}
	
	// locate all i2c host controllers
	Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiI2cHostProtocolGuid, NULL, &HandleCount, &HandleBuffer);
	if (EFI_ERROR(Status)) {
		// no i2c host controller found
		_print(SystemTable, (UINT32)0xFFFFFFFF, 0, 0, 0, 0, 0);
		CpuDeadLoop();
	}
	
	// get the first availiable i2c controller
	Status = gBS->HandleProtocol(HandleBuffer[0], &gEfiI2cHostProtocolGuid, (VOID **)I2cHost);
	if (EFI_ERROR(Status)) {
		// failed to access
		_print(SystemTable, (UINT32)0xFFFFFFF1, 0, 0, 0, 0, 0);
		CpuDeadLoop();
	} else {
		_print(SystemTable, (UINT32)0x1234, 0, 0, 0, 0, 0);
	}
	
	return Status;
	
}

/* SMBus Host controller */
EFI_STATUS FindSMBusHostController(IN EFI_SYSTEM_TABLE  *SystemTable, EFI_SMBUS_HC_PROTOCOL **Smbus)
{
	EFI_STATUS Status;
	EFI_HANDLE *HandleBuffer;
	UINTN HandleCount;
	
	Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiSmbusHcProtocolGuid, NULL, &HandleCount, &HandleBuffer);
	if (EFI_ERROR(Status)) {
		// no i2c host controller found
		_print(SystemTable, (UINT32)0x1FFFFFFF, 0, 0, 0, 0, 0);
		CpuDeadLoop();
	} else {
		_print(SystemTable, (UINT32)0x2FFFFFFF, 0, 0, 0, 0, 0);
	}
	
	Status = gBS->HandleProtocol(HandleBuffer[0], &gEfiSmbusHcProtocolGuid, (VOID **)Smbus);
	if (EFI_ERROR(Status)) {
		// no i2c host controller found
		_print(SystemTable, (UINT32)0x1aFFFFFF, 0, 0, 0, 0, 0);
		CpuDeadLoop();
	} else {
		// smbus host controller found
		_print(SystemTable, (UINT32)0x3FFFFFFF, 0, 0, 0, 0, 0);
	}
	
	return Status;
	
}

#define SMBUS_MIN_ADDR 0x03 // lowest possible SMBus address
#define SMBUS_MAX_ADDR 0x77 // highest possible SMbus address

/* Function to scan SMBus for display devices */
// yes, that's correct *Smbus not **Smbus in parameter for Smbus->execute calls
EFI_STATUS ScanBus(IN EFI_SYSTEM_TABLE  *SystemTable, EFI_SMBUS_HC_PROTOCOL *Smbus)
{
	EFI_STATUS Status;
	static EFI_SMBUS_DEVICE_ADDRESS smbus_addr;
	EFI_SMBUS_DEVICE_COMMAND command = 0x00;
	
	UINTN length = 1;
	UINT8 data;
	
	// scanning bus for connected device code 
	_print(SystemTable, (UINT32)0xFF1FFFFF, 0, 0, 0, 0, 0);
	
	
	for (UINT8 Address = SMBUS_MIN_ADDR; Address <= SMBUS_MAX_ADDR; Address++) {
		Status = Smbus->Execute(Smbus, smbus_addr, command, EfiSmbusReadByte, FALSE, &length, &data);
		smbus_addr.SmbusDeviceAddress = Address;
		if (EFI_ERROR(Status)) {
			_print(SystemTable, (UINT32)0xabcdef00, 0, 0, 0, 0, 1);
		} else {
		_print(SystemTable, (UINT32)Address, 0, 0, 0, 0, 1);
		_print(SystemTable, (UINT32)length, 0, 0, 0, 0, 1);
		_print(SystemTable, (UINT32)data, 0, 0, 0, 0, 1);
		}
	}
	
	// scan complete info
	_print(SystemTable, (UINT32)0xFF1111FF, 0, 0, 0, 0, 0);
	
	/*
	 ---- do test for this --- not working -----
	// read block of data - UDID
	EFI_SMBUS_UDID udid;
	UINTN len2 = sizeof(EFI_SMBUS_UDID);
	
	for (UINT8 Address = SMBUS_MIN_ADDR; Address <= 0x7F; Address++) {
		Status = Smbus->Execute(Smbus, smbus_addr, command, EfiSmbusReadBlock, FALSE, &len2, &udid);
		smbus_addr.SmbusDeviceAddress = Address;
		if (EFI_ERROR(Status)) {
			_print(SystemTable, (UINT32)0xabcdef00, 0, 0, 0, 0, 1);
		} else {
		_print(SystemTable, (UINT32)Address, 0, 0, 0, 0, 1);
		_print(SystemTable, (UINT32)len2, 0, 0, 0, 0, 1);
		_print(SystemTable, (UINT32)udid.VendorSpecificId, 0, 0, 0, 0, 1);
		}
	}
	
	// scan complete info
	_print(SystemTable, (UINT32)0xFF111bFF, 0, 0, 0, 0, 0);
	*/
	
	return EFI_SUCCESS;
}

// definition below FindUsbHostController
EFI_STATUS ReadUsbDeviceDescriptor(IN EFI_SYSTEM_TABLE  *SystemTable, EFI_USB2_HC_PROTOCOL *usb, UINT8 deviceAddress);

/* USB Host Controller */
EFI_STATUS FindUsbHostController(IN EFI_SYSTEM_TABLE  *SystemTable, EFI_USB2_HC_PROTOCOL **usb)
{
	EFI_STATUS Status;
	EFI_HANDLE *HandleBuffer;
	UINTN HandleCount;
	UINT8 NumPorts;
	static EFI_USB2_HC_PROTOCOL *usb2;
	UINT8 maxSpeed, is64;
	
	Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HandleCount, &HandleBuffer);
	if (EFI_ERROR(Status)) {
		// host controller not found
		_print(SystemTable, (UINT32)0x1FFFFFFF, 0, 0, 0, 0, 0);
		CpuDeadLoop();
	} else {
		// found - pass test
		_print(SystemTable, (UINT32)0x2FFFFFFF, 0, 0, 0, 0, 0);
	}
	
	// print handle count if succeed
	_print(SystemTable, (UINT32)HandleCount, 0, 0, 0, 0, 0);
		
	for (UINTN i = 0; i < HandleCount; i++) {
		
		Status = gBS->HandleProtocol(HandleBuffer[i], &gEfiUsb2HcProtocolGuid, (VOID **)&usb2);
		if (EFI_ERROR(Status)) {
			// failed to get EFI_USB2_HC_PROTOCOL
			_print(SystemTable, (UINT32)0x1aFFFFFF, 0, 0, 0, 0, 1);
		} else {
			// ok pass test
			_print(SystemTable, (UINT32)0x3FFFFFFF, 0, 0, 0, 0, 1);
		}
		
		Status = usb2->GetCapability(usb2, &maxSpeed, &NumPorts, &is64);
		if (EFI_ERROR(Status)) {
			// ERROR getting USB controller capabilities
			_print(SystemTable, (UINT32)0x1bFFFFFF, 0, 0, 0, 0, 1);
		} else {
			// usb controller max speed, ports, 64-bit
			_print(SystemTable, (UINT32)i, 0, 0, 0, 0, 1);
			_print(SystemTable, (UINT32)maxSpeed, 0, 0, 0, 0, 1);
			_print(SystemTable, (UINT32)NumPorts, 0, 0, 0, 0, 1);
			_print(SystemTable, (UINT32)is64, 0, 0, 0, 0, 1);
			_print(SystemTable, (UINT32)Status, 0, 0, 0, 0, 0);
		}
		
		// if pass this execute 
		//ReadUsbDeviceDescriptor(SystemTable, usb, NumPorts);
		
		// iterate through each USB port
		for (UINT8 port = 0; port < NumPorts; port++) {
			EFI_USB_PORT_STATUS PortStatus;
			Status = usb2->GetRootHubPortStatus(usb2, port, &PortStatus);
			if (EFI_ERROR(Status)) {
				// ERROR 
				_print(SystemTable, (UINT32)0x1cFFFFFF, 0, 0, 0, 0, 1);
			} else {
				// usb controller max speed, ports, 64-bit
				_print(SystemTable, (UINT32)port, 0, 0, 0, 0, 1);
				_print(SystemTable, (UINT32)PortStatus.PortStatus, 0, 0, 0, 0, 0);
			}
		}
		
	}
	
	return Status;
	
}

/*
typedef struct {
	UINT8	TranslatorHubAddress;
	UINT8	TranslatorPortNumber;
} EFI_USB2_HC_TRANSACTION_TRANSLATOR;
*/

/* function to read USB device descriptor */
EFI_STATUS ReadUsbDeviceDescriptor(IN EFI_SYSTEM_TABLE  *SystemTable, EFI_USB2_HC_PROTOCOL *usb, UINT8 deviceAddress)
{
	
	EFI_STATUS Status;
	EFI_USB_DEVICE_DESCRIPTOR devDesc;
	EFI_USB_DEVICE_REQUEST request;
	UINT32 timeout = 5000; // 5 sec
	//UINTN dataLength = sizeof(EFI_USB_DEVICE_DESCRIPTOR);
	UINTN dataLength;
	UINT32 result;
	EFI_USB2_HC_TRANSACTION_TRANSLATOR *translator;
	
	request.RequestType = 0x80; // device to host | standard | device
	request.Request = 0x06; // get descriptor
	request.Value = 0x0100; // device descriptor
	request.Index = 0;
	request.Length = sizeof(EFI_USB_DEVICE_DESCRIPTOR);
	
	Status = usb->ControlTransfer(usb, deviceAddress, 2, 64, &request, EfiUsbDataIn,
		&devDesc, &dataLength, timeout, translator, &result);
	
	if (EFI_ERROR(Status)) {
			// failed to get ControlTransfer
			_print(SystemTable, (UINT32)0x12FFFFFF, 0, 0, 0, 0, 1);
		} else {
			// ok pass test
			_print(SystemTable, (UINT32)devDesc.IdVendor, 0, 0, 0, 0, 1);
			_print(SystemTable, (UINT32)devDesc.IdProduct, 0, 0, 0, 0, 1);
			_print(SystemTable, (UINT32)devDesc.DeviceClass, 0, 0, 0, 0, 0);
		}
	
	return EFI_SUCCESS;
}

/**
  The user Entry Point for Application. The user code starts with this function
  as the real entry point for the application.

  @param[in] ImageHandle    The firmware allocated handle for the EFI image.
  @param[in] SystemTable    A pointer to the EFI System Table.

  @retval EFI_SUCCESS       The entry point is executed successfully.
  @retval other             Some error occurs when executing this entry point.

**/
EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
	
	EFI_STATUS Status;
	
				/* ---------------------------------------------------------------------- */
				// USB protocol demo
				/* ---------------------------------------------------------------------- */
	
#if 1

	static EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *PciRootBridgeIo = NULL;
	UINTN HandleCount;
	EFI_HANDLE *HandleBuffer;
	
	Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciRootBridgeIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);

	if (HandleCount == 0) {
		// nie mam zadnych bledow na emulatorze, czyli samo modul zaladowalo chyba do tego momentu poprawnie
		DEBUG ((DEBUG_INFO, " return val for count %d\n", Status));
	}
	
	Status = gBS->HandleProtocol(HandleBuffer[0], &gEfiPciRootBridgeIoProtocolGuid, (VOID **)&PciRootBridgeIo);
	
	// setup GOP
	Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&mGraphicsOuput);
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	UINTN gop_querymode_size = sizeof(EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
	EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *mode_info = NULL;
	Status = mGraphicsOuput->QueryMode(mGraphicsOuput, mGraphicsOuput->Mode->Mode, 
		&gop_querymode_size, &mode_info); 
	
	// ta funkcja to sobie moge na GOP Emulator x64 ale nie na prawdziwym sprzecie... to tak btw.
	if (mode_info == NULL) {
		DEBUG ((DEBUG_INFO, " mode info erro %d\n", Status));
	}
	
	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
	
	//Status = gBS->HandleProtocol
	_print(SystemTable, (UINT32)HandleCount, 0, 0, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)&HandleBuffer[0], 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)HandleBuffer, 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)&PciRootBridgeIo, 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)PciRootBridgeIo, 1, 0, 0, 0);
	
	// set GOP mode to 0 = 1024x768  |  1 = 800x600
	mGraphicsOuput->SetMode(mGraphicsOuput, 0);
	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	
	// code here...
	EFI_USB2_HC_PROTOCOL *usb2hc;
	FindUsbHostController(SystemTable, &usb2hc);
	
	
	// wait for key event instead of continue shell 
	EFI_INPUT_KEY key;
	while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS) {}
	
	// shutdown procedure
	SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
	
	return EFI_SUCCESS;
	
#endif	
	
	
				/* ---------------------------------------------------------------------- */
				// I2C demo
				/* ---------------------------------------------------------------------- */

#if 0

	static EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *PciRootBridgeIo = NULL;
	UINTN HandleCount;
	EFI_HANDLE *HandleBuffer;
	
	Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciRootBridgeIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);

	if (HandleCount == 0) {
		// nie mam zadnych bledow na emulatorze, czyli samo modul zaladowalo chyba do tego momentu poprawnie
		DEBUG ((DEBUG_INFO, " return val for count %d\n", Status));
	}
	
	Status = gBS->HandleProtocol(HandleBuffer[0], &gEfiPciRootBridgeIoProtocolGuid, (VOID **)&PciRootBridgeIo);
	
	// setup GOP
	Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&mGraphicsOuput);
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	UINTN gop_querymode_size = sizeof(EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
	EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *mode_info = NULL;
	Status = mGraphicsOuput->QueryMode(mGraphicsOuput, mGraphicsOuput->Mode->Mode, 
		&gop_querymode_size, &mode_info); 
	
	// ta funkcja to sobie moge na GOP Emulator x64 ale nie na prawdziwym sprzecie... to tak btw.
	if (mode_info == NULL) {
		DEBUG ((DEBUG_INFO, " mode info erro %d\n", Status));
	}
	
	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
	
	//Status = gBS->HandleProtocol
	_print(SystemTable, (UINT32)HandleCount, 0, 0, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)&HandleBuffer[0], 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)HandleBuffer, 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)&PciRootBridgeIo, 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)PciRootBridgeIo, 1, 0, 0, 0);
	
	// set GOP mode to 0 = 1024x768  |  1 = 800x600
	mGraphicsOuput->SetMode(mGraphicsOuput, 0);
	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	
	// code here...
	//EFI_I2C_HOST_PROTOCOL *I2cHost;
	//Status = FindI2cHostController(SystemTable, &I2cHost);
	//if (EFI_ERROR(Status)) {
	//	_print(SystemTable, (UINT32)0xFFFFFFF2, 0, 0, 0, 0, 0);
	//}
	
	// i2c returned 0xFFFFFFFF code from this implementation = not found any host devices
	EFI_SMBUS_HC_PROTOCOL *Smbus;
	Status = FindSMBusHostController(SystemTable, &Smbus);
	ScanBus(SystemTable, Smbus);
	
	
	// wait for key event instead of continue shell 
	EFI_INPUT_KEY key;
	while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS) {}
	
	// shutdown procedure
	SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
	
	return EFI_SUCCESS;
	
#endif

				/* ---------------------------------------------------------------------- */
				// PCI demo
				/* ---------------------------------------------------------------------- */

#if 0

	static EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *PciRootBridgeIo = NULL;
	UINTN HandleCount;
	EFI_HANDLE *HandleBuffer;
	
	Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciRootBridgeIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);

	if (HandleCount == 0) {
		// nie mam zadnych bledow na emulatorze, czyli samo modul zaladowalo chyba do tego momentu poprawnie
		DEBUG ((DEBUG_INFO, " return val for count %d\n", Status));
	}
	
	Status = gBS->HandleProtocol(HandleBuffer[0], &gEfiPciRootBridgeIoProtocolGuid, (VOID **)&PciRootBridgeIo);
	
	// setup GOP
	Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&mGraphicsOuput);
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	UINTN gop_querymode_size = sizeof(EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
	EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *mode_info = NULL;
	Status = mGraphicsOuput->QueryMode(mGraphicsOuput, mGraphicsOuput->Mode->Mode, 
		&gop_querymode_size, &mode_info); 
	
	// ta funkcja to sobie moge na GOP Emulator x64 ale nie na prawdziwym sprzecie... to tak btw.
	if (mode_info == NULL) {
		DEBUG ((DEBUG_INFO, " mode info erro %d\n", Status));
	}
	
	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
	
	//Status = gBS->HandleProtocol
	_print(SystemTable, (UINT32)HandleCount, 0, 0, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)&HandleBuffer[0], 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)HandleBuffer, 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)&PciRootBridgeIo, 1, 0, 0, 0);
	_print(SystemTable, 0, (UINT64)PciRootBridgeIo, 1, 0, 0, 0);
	
	// enumerate pci bridge - 
	// - default for demo BUS = 256 but in real hardware is max 2 buses.
	// for test set to 5 
	for (UINT8 Bus = 0; Bus < 5; Bus++) {
		for (UINT8 Device = 0; Device < 32; Device++) {
			for (UINT8 Function = 0; Function < 8; Function++) {
				UINT32 VendorDeviceId;
				Status = ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x00, &VendorDeviceId);
				if (EFI_ERROR(Status)) {
					_print(SystemTable, 0xFFFFFFFF, 0, 0, 0, 0, 0);
				}
				
				// check if the device exists
				if ((VendorDeviceId & 0xFFFF) != 0xFFFF) {
					UINT16 VendorId = VendorDeviceId & 0xFFFF;
					UINT16 DeviceId = (VendorDeviceId >> 16) & 0xFFFF;
	
					// check memory mapped for XEON 1200 - this is CPU internal mechanism for PCI
					// needs to find this device id and get all reources
					// DRAM controller device 0, function 0, bus 0
					if (DeviceId == 0x0104){
						
					_print(SystemTable, (UINT32)Bus, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)Device, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)Function, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)VendorId, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)DeviceId, 0, 0, 0, 0, 0);
					
					// check the device class code
					UINT32 ClassCodeReg;
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x08, &ClassCodeReg);
					UINT8 BaseClass = (ClassCodeReg >> 24) & 0xFF;
					UINT8 SubClass = (ClassCodeReg >> 16) & 0xFF;
					_print(SystemTable, (UINT32)BaseClass, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)SubClass, 0, 0, 0, 0, 0);
					
					
					// get BAR
					UINT32 Bar0;
					UINT32 Bar1;
					UINT32 Bar2;
					UINT32 Bar3;
					UINT32 Bar4;
					UINT32 Bar5;
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x10, &Bar0);
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x14, &Bar1);
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x18, &Bar2);
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x1c, &Bar3);
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x20, &Bar4);
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x24, &Bar5);
					_print(SystemTable, (UINT32)Bar0, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)Bar1, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)Bar2, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)Bar3, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)Bar4, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)Bar5, 0, 0, 0, 0, 1); // nie rob nowej linii
					
					// read interrupt line and pin
					UINT32 Reg3c; // offset 0x3C w configuration space header
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x3c, &Reg3c);
					// pobierz caly vector, extrakcje zrobie w pythonie potem
					_print(SystemTable, (UINT32)Reg3c, 0, 0, 0, 0, 0);
					
					// Host Bridge / DRAM Registers
					UINT32 pciConfig; // PCICMD - 16 bits
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x4, &pciConfig);
					UINT32 pciStatus; // PCISTS - 16 bits
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x6, &pciStatus);
					
					_print(SystemTable, (UINT32)pciConfig, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)pciStatus, 0, 0, 0, 0, 1);
					
					// 64 bit register
					UINT64 pxpepbar;
					ReadPciConfig64(PciRootBridgeIo, Bus, Device, Function, 0x40, &pxpepbar);
					
					_print(SystemTable, 0, (UINT64)pxpepbar, 0, 0, 0, 0);
					
					// I need some infor about GMCH register
					UINT32 gmch;
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x50, &gmch);
					_print(SystemTable, (UINT32)gmch, 0, 0, 0, 0, 0);
					
					UINT32 deven;
					ReadPciConfig(PciRootBridgeIo, Bus, Device, Function, 0x54, &deven);
					_print(SystemTable, (UINT32)deven, 0, 0, 0, 0, 0);
					
					// end 
					}

				}
			}
		}
	}
	
	// wait for key event instead of continue shell 
	EFI_INPUT_KEY key;
	while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS) {}
	
	// shutdown procedure
	SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
	
	return EFI_SUCCESS;
	
#endif
				/* ---------------------------------------------------------------------- */
				// GOP - ale na HP to nie dziala
				// sprawdzilem na windows 10 jakie sa adresy dla Intel 4600 no i jednak sa inne niz u mnie
				// dlatego wersja z PCI bedzie sprawadzana teraz
				/* ---------------------------------------------------------------------- */

#if 0
	// graphics protocol test
	Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&mGraphicsOuput);
	//DEBUG((DEBUG_INFO, "GOP status %r \n", Status));
	
	//mGraphicsOuput->SetMode(mGraphicsOuput, 4);
	
	// reset currsor position
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	
	// gop queryMode
	UINTN gop_querymode_size = sizeof(EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
	EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *mode_info = NULL;
	Status = mGraphicsOuput->QueryMode(mGraphicsOuput, mGraphicsOuput->Mode->Mode, 
		&gop_querymode_size, &mode_info); 
	
	if (mode_info == NULL) {
		DEBUG ((DEBUG_INFO, " mode info erro %d\n", Status));
	}
	
	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
	
	//UINT32 _maxMode = (UINT32)mGraphicsOuput->Mode->MaxMode;
	//_print(SystemTable, _maxMode, 0, 0, 0);
	
	_print(SystemTable, mGraphicsOuput->Mode->MaxMode, 0, 0, 0, 0, 0);
	_print(SystemTable, mGraphicsOuput->Mode->Mode, 0, 0, 0, 0, 0);
	_print(SystemTable, mode_info->HorizontalResolution, 0, 0, 0, 0, 0);
	_print(SystemTable, mode_info->VerticalResolution, 0, 0, 0, 0, 0);
	_print(SystemTable, 0, mGraphicsOuput->Mode->FrameBufferBase, 1, 0, 0, 0); // EFI_PHYSICAL_ADDRESS is 64
	_print(SystemTable, 0, mGraphicsOuput->Mode->FrameBufferSize, 1, 0, 0, 0);
	_print(SystemTable, mode_info->PixelFormat, 0, 0, 0, 0, 0);
	_print(SystemTable, mode_info->PixelsPerScanLine, 0, 0, 0, 0, 0);
	
	for (INTN i = 0; i < mGraphicsOuput->Mode->MaxMode; i++) {
		 mGraphicsOuput->QueryMode(mGraphicsOuput, (UINT32)i, &gop_querymode_size, &mode_info);
		_print(SystemTable, (UINT32)i, 0, 0, 0, 0, 1);
		_print(SystemTable, mode_info->HorizontalResolution, 0, 0, 0, 0, 1);
		_print(SystemTable, mode_info->VerticalResolution, 0, 0, 0, 0, 0);
	}
	
	UINT8 r1 = ReadVgaCrtReg(0x00);
	UINT8 r2 = ReadVgaCrtReg(0x01);
	UINT8 r3 = ReadVgaCrtAttr(0x00);
	UINT8 r4 = ReadVgaCrtAttr(0x01);
	_print(SystemTable, (UINT32)r1, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r4, 0, 0, 0, 0, 0);
	
	// #define VGA_GRX 0x3ce 
	// #define VGA_CR10_BASE 0x3cf // this is index 10
	UINT8 r20 = ReadGRX(0x05); // GR05 - Graphics mode register
	UINT8 r21 = ReadGRX(0x10); // GR10 - address mapping
	UINT8 r22 = ReadGRX(0x11); // GR11 - Page Selector
	_print(SystemTable, (UINT32)r20, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r21, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r22, 0, 0, 0, 0, 0);
	
	// read VGA settings - full detail
	// horizontal timing
	UINT8 r30 = ReadVga(0x00);
	UINT8 r31 = ReadVga(0x01);
	UINT8 r32 = ReadVga(0x02);
	UINT8 r33 = ReadVga(0x03);
	UINT8 r34 = ReadVga(0x04);
	UINT8 r35 = ReadVga(0x05);
	
	// vertical timing
	UINT8 r36 = ReadVga(0x06);
	UINT8 r37 = ReadVga(0x07);
	UINT8 r38 = ReadVga(0x12);
	UINT8 r39 = ReadVga(0x10);
	UINT8 r40 = ReadVga(0x11);
	UINT8 r41 = ReadVga(0x15);
	UINT8 r42 = ReadVga(0x16);
	
	// decode overflows bits from reg 0x07
	UINT16 f1 = ((r37 & 0x01) << 8) | r36;
	UINT16 f2 = ((r37 & 0x02) << 7) | r38;
	UINT16 f3 = ((r37 & 0x04) << 6) | r39;
	UINT16 f4 = ((r37 & 0x08) << 5) | r41;
	
	// read pixel clock division and other settings
	UINT8 ModeControl = ReadVga(0x17);
	BOOLEAN isInterlaced = (ModeControl & 0x80) != 0;
	
	// print all
	_print(SystemTable, (UINT32)r30, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r31, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r32, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r33, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r34, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r35, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r36, 0, 0, 0, 0, 0);
	_print(SystemTable, (UINT32)r37, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r38, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r39, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r40, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r41, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r42, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f1, 0, 0, 0, 0, 0);
	_print(SystemTable, (UINT32)f2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f4, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)ModeControl, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)isInterlaced, 0, 0, 0, 0, 0);
	
	/* update 29.01.2025 */
	
	
	// msr misc output 0x3c2
	UINT8 MsrVal1 = IoRead8(0x3c2);
	UINT8 MsrVal2 = IoRead8(0x3cc);
	
	UINT32 mmioBase = 0xe000;
	
	UINT16 devVal1 = IoRead16(0x54);
	UINT16 devVal2 = IoRead16(0x50);
	
	UINT16 devVal3 = IoRead16(mmioBase+ 0x54);
	UINT16 devVal4 = IoRead16(mmioBase+ 0x50);
	
	_print(SystemTable, (UINT32)devVal1, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)devVal2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)devVal3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)devVal4, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)MsrVal2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)MsrVal1, 0, 0, 0, 0, 0);
	
	//while(1);
	
	/* END of section - update 29-01-2025 */
	
	
	
	CHAR16 buffer[256]; // eax  
	CHAR16 buffer1[256]; // eax  
  CHAR16 *hexchar = L"0123456789ABCDEF";
  
    UINT32 state1 = 0;
  
	UINT64 tmpVal2 = 0;
	for (UINT64 i1 = 0; i1 < 100000000; i1++) {
		for (UINT64 j1 = 0; j1 < 100000000; j1++) {
			for (UINT64 k1 = 0; k1 < 100000000; k1++) {
				tmpVal2 += 1;
				
				SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 40, 10);
				for (INTN j = 0; j < 16; j++) {
					buffer[15 - j] = hexchar[(tmpVal2 >> (j * 4)) & 0xf];
				  }
				  buffer[16] = L'\r'; buffer[17] = L'\n'; buffer[18] = L'\0'; 
				  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
				  
				  // state
				SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 40, 11);
				for (INTN j = 0; j < 16; j++) {
					buffer1[15 - j] = hexchar[(state1 >> (j * 4)) & 0xf];
				  }
				  buffer1[16] = L'\r'; buffer1[17] = L'\n'; buffer1[18] = L'\0'; 
				  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer1);
				  
				// if 
				if (tmpVal2 > 0x8000) {
					tmpVal2 = 0;
					
					
					if (state1 == 1){
						state1 = 0;
					} else if (state1 == 0) {
						state1 = 1;
					}
					
					
					// set mode to 0 and print all again
					mGraphicsOuput->SetMode(mGraphicsOuput, state1);
					
					mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
					
					SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
					
					r1 = ReadVgaCrtReg(0x00);
					r2 = ReadVgaCrtReg(0x01);
					r3 = ReadVgaCrtAttr(0x00);
					r4 = ReadVgaCrtAttr(0x01);
					_print(SystemTable, (UINT32)r1, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r2, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r3, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r4, 0, 0, 0, 0, 0);
					
					// #define VGA_GRX 0x3ce 
					// #define VGA_CR10_BASE 0x3cf // this is index 10
					r20 = ReadGRX(0x05); // GR05 - Graphics mode register
					r21 = ReadGRX(0x10); // GR10 - address mapping
					r22 = ReadGRX(0x11); // GR11 - Page Selector
					_print(SystemTable, (UINT32)r20, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r21, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r22, 0, 0, 0, 0, 0);
					
					// read VGA settings - full detail
					// horizontal timing
					r30 = ReadVga(0x00);
					r31 = ReadVga(0x01);
					r32 = ReadVga(0x02);
					r33 = ReadVga(0x03);
					r34 = ReadVga(0x04);
					r35 = ReadVga(0x05);
					
					// vertical timing
					r36 = ReadVga(0x06);
					r37 = ReadVga(0x07);
					r38 = ReadVga(0x12);
					r39 = ReadVga(0x10);
					r40 = ReadVga(0x11);
					r41 = ReadVga(0x15);
					r42 = ReadVga(0x16);
					
					// decode overflows bits from reg 0x07
					f1 = ((r37 & 0x01) << 8) | r36;
					f2 = ((r37 & 0x02) << 7) | r38;
					f3 = ((r37 & 0x04) << 6) | r39;
					f4 = ((r37 & 0x08) << 5) | r41;
					
					// read pixel clock division and other settings
					ModeControl = ReadVga(0x17);
					isInterlaced = (ModeControl & 0x80) != 0;
					
					// print all
					_print(SystemTable, (UINT32)r30, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r31, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r32, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r33, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r34, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r35, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r36, 0, 0, 0, 0, 0);
					_print(SystemTable, (UINT32)r37, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r38, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r39, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r40, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r41, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)r42, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)f1, 0, 0, 0, 0, 0);
					_print(SystemTable, (UINT32)f2, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)f3, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)f4, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ModeControl, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)isInterlaced, 0, 0, 0, 0, 0);
					
					
					// 29.01.2025
					// read all memory from intel 3000 hd addresses
					UINT32 addr1 = 0xdd400000;
					UINT32 addr2 = 0xb0000000;
					UINT32 addr3 = 0xe000;
					UINT32 addr4 = 0x3b0;
					UINT32 addr5 = 0x3c0;
					UINT32 addr6 = 0xa0000;
					
					UINT16 ddd1 = IoRead16(addr1 + 0x54);
					UINT16 ddd2 = IoRead16(addr1 + 0x50);
					
					UINT16 ddd3 = IoRead16(addr2 + 0x54);
					UINT16 ddd4 = IoRead16(addr2 + 0x50);
					
					UINT16 ddd5 = IoRead16(addr3 + 0x54);
					UINT16 ddd6 = IoRead16(addr3 + 0x50);
					
					UINT16 ddd7 = IoRead16(addr4 + 0x54);
					UINT16 ddd8 = IoRead16(addr4 + 0x50);
					
					UINT16 ddd9 = IoRead16(addr5 + 0x54);
					UINT16 ddd10 = IoRead16(addr5 + 0x50);
					
					UINT16 ddd11 = IoRead16(addr6 + 0x54);
					UINT16 ddd12 = IoRead16(addr6 + 0x50);
					
					_print(SystemTable, (UINT32)ddd1, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd2, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd3, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd4, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd5, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd6, 0, 0, 0, 0, 0);
					
					_print(SystemTable, (UINT32)ddd7, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd8, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd9, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd10, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd11, 0, 0, 0, 0, 1);
					_print(SystemTable, (UINT32)ddd12, 0, 0, 0, 0, 0);
					// end 29.01.2025
					
					
				}
			}
		}
	}
	
	// set mode to 0 and print all again
	mGraphicsOuput->SetMode(mGraphicsOuput, 0);
	
	r1 = ReadVgaCrtReg(0x00);
	r2 = ReadVgaCrtReg(0x01);
	r3 = ReadVgaCrtAttr(0x00);
	r4 = ReadVgaCrtAttr(0x01);
	_print(SystemTable, (UINT32)r1, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r4, 0, 0, 0, 0, 0);
	
	// #define VGA_GRX 0x3ce 
	// #define VGA_CR10_BASE 0x3cf // this is index 10
	r20 = ReadGRX(0x05); // GR05 - Graphics mode register
	r21 = ReadGRX(0x10); // GR10 - address mapping
	r22 = ReadGRX(0x11); // GR11 - Page Selector
	_print(SystemTable, (UINT32)r20, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r21, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r22, 0, 0, 0, 0, 0);
	
	// read VGA settings - full detail
	// horizontal timing
	r30 = ReadVga(0x00);
	r31 = ReadVga(0x01);
	r32 = ReadVga(0x02);
	r33 = ReadVga(0x03);
	r34 = ReadVga(0x04);
	r35 = ReadVga(0x05);
	
	// vertical timing
	r36 = ReadVga(0x06);
	r37 = ReadVga(0x07);
	r38 = ReadVga(0x12);
	r39 = ReadVga(0x10);
	r40 = ReadVga(0x11);
	r41 = ReadVga(0x15);
	r42 = ReadVga(0x16);
	
	// decode overflows bits from reg 0x07
	f1 = ((r37 & 0x01) << 8) | r36;
	f2 = ((r37 & 0x02) << 7) | r38;
	f3 = ((r37 & 0x04) << 6) | r39;
	f4 = ((r37 & 0x08) << 5) | r41;
	
	// read pixel clock division and other settings
	ModeControl = ReadVga(0x17);
	isInterlaced = (ModeControl & 0x80) != 0;
	
	// print all
	_print(SystemTable, (UINT32)r30, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r31, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r32, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r33, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r34, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r35, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r36, 0, 0, 0, 0, 0);
	_print(SystemTable, (UINT32)r37, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r38, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r39, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r40, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r41, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r42, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f1, 0, 0, 0, 0, 0);
	_print(SystemTable, (UINT32)f2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f4, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)ModeControl, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)isInterlaced, 0, 0, 0, 0, 0);
	
	/*
	UINT64 tmpVal3 = 0;
	for (UINT64 i2 = 0; i2 < 100000000; i2++) {
		for (UINT64 j2 = 0; j2 < 100000000; j2++) {
			for (UINT64 k2 = 0; k2 < 100000000; k2++) {
				tmpVal3 += 1;
			}
		}
	}
	
	
	// set mode to 0 and print all again
	mGraphicsOuput->SetMode(mGraphicsOuput, 0);
	r1 = ReadVgaCrtReg(0x00);
	r2 = ReadVgaCrtReg(0x01);
	r3 = ReadVgaCrtAttr(0x00);
	r4 = ReadVgaCrtAttr(0x01);
	_print(SystemTable, (UINT32)r1, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r4, 0, 0, 0, 0, 0);
	
	// #define VGA_GRX 0x3ce 
	// #define VGA_CR10_BASE 0x3cf // this is index 10
	r20 = ReadGRX(0x05); // GR05 - Graphics mode register
	r21 = ReadGRX(0x10); // GR10 - address mapping
	r22 = ReadGRX(0x11); // GR11 - Page Selector
	_print(SystemTable, (UINT32)r20, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r21, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r22, 0, 0, 0, 0, 0);
	
	// read VGA settings - full detail
	// horizontal timing
	r30 = ReadVga(0x00);
	r31 = ReadVga(0x01);
	r32 = ReadVga(0x02);
	r33 = ReadVga(0x03);
	r34 = ReadVga(0x04);
	r35 = ReadVga(0x05);
	
	// vertical timing
	r36 = ReadVga(0x06);
	r37 = ReadVga(0x07);
	r38 = ReadVga(0x12);
	r39 = ReadVga(0x10);
	r40 = ReadVga(0x11);
	r41 = ReadVga(0x15);
	r42 = ReadVga(0x16);
	
	// decode overflows bits from reg 0x07
	f1 = ((r37 & 0x01) << 8) | r36;
	f2 = ((r37 & 0x02) << 7) | r38;
	f3 = ((r37 & 0x04) << 6) | r39;
	f4 = ((r37 & 0x08) << 5) | r41;
	
	// read pixel clock division and other settings
	ModeControl = ReadVga(0x17);
	isInterlaced = (ModeControl & 0x80) != 0;
	
	// print all
	_print(SystemTable, (UINT32)r30, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r31, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r32, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r33, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r34, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r35, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r36, 0, 0, 0, 0, 0);
	_print(SystemTable, (UINT32)r37, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r38, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r39, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r40, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r41, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r42, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f1, 0, 0, 0, 0, 0);
	_print(SystemTable, (UINT32)f2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)f4, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)ModeControl, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)isInterlaced, 0, 0, 0, 0, 0);
	
	*/
	
	// read CRT registers
	/*
	static UINTN VGA_ADDR_PORT = 0x3d4;
	static UINTN VGA_DATA_PORT = 0x3d5;
	UINT32 r1 = IoRead32(VGA_ADDR_PORT);
	UINT32 r2 = IoRead32(VGA_DATA_PORT);
	UINT8 r3 = IoRead8(VGA_ADDR_PORT);
	UINT8 r4 = IoRead8(VGA_DATA_PORT);
	_print(SystemTable, (UINT32)r1, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r2, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r3, 0, 0, 0, 0, 1);
	_print(SystemTable, (UINT32)r4, 0, 0, 0, 0, 0);
	*/
	
	
	EFI_INPUT_KEY key;
	while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS) {}
	
	// shutdown procedure
	SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
	
	/*
	EFI_INPUT_KEY key;
	while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS) {
		mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
		//MicroSecondDelay(9000000);
		UINT32 temp = 0;
		for (INTN i = 0; i < 0xffffffff - 1; i++) {
			for (INTN j = 0; j < 1000000; j++) {
				temp++;
			}
		}
		
		mGraphicsOuput->Blt(mGraphicsOuput, &blue, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
		//MicroSecondDelay(9000000);
		

	}
	
	// shutdown procedure
	SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
	*/
	
	//CpuDeadLoop();
	
	return EFI_SUCCESS;

#endif

// --- demo z dnia 14-01-2025 
// wyswietlane sa tutaj informacje z pliku invd.asm i mix tego na ekranie z rejestrow MSR
// ale dla testow z GOP poniewaz nie moge zrobić nowego repo jak test0x100.informacje
// po prostu zakomentuje i wylacze z kompilacji na potrzeby nauki innych protokolow jak GOP
#if 0
	
	// gEfiGraphicsOutputProtocolGuid
	
	//CpuBreakpoint();
	//DEBUG ((DEBUG_INFO, " [ BIOS test ] demo1 1 %d\n", ImageHande));
	//CHAR8 buf1[64];
	////ValueToHexStr(buf1, ImageHandle, NULL, sizoef(ImageHandle));

  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_YELLOW, EFI_GREEN));
  SystemTable->ConOut->ClearScreen(SystemTable->ConOut);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"Hello, World!\r\n\r\n");
  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_RED, EFI_BLACK));
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"Press any key to shutdown");
  
  UINT32 eax, ebx, ecx, edx;
  //AsmCpuid(0x06, &eax, &ebx, &ecx, &edx);
  
  AsmCpuid(0x01, &eax, &ebx, &ecx, &edx);
  
  CHAR16 buffer[64]; // eax
  CHAR16 ebxb[64];
  CHAR16 ecxb[64];
  CHAR16 edxb[64];
  // espb[64];
  //CHAR16 rbpb[64];
  //UnicodeSPrint(buffer, sizeof(buffer), L"eax: 0x%08x\n", eax);
  //UnicodeValueToString(buffer, LEFT_JUSTIFY, eax, 16);
  
  CHAR16 *hexchar = L"0123456789ABCDEF";
  
  for (INTN i = 0; i < 8; i++) {
	buffer[7 - i] = hexchar[(eax >> (i * 4)) & 0xf];
	ebxb[7 - i] = hexchar[(ebx >> (i * 4)) & 0xf];
	ecxb[7 - i] = hexchar[(ecx >> (i * 4)) & 0xf];
	edxb[7 - i] = hexchar[(edx >> (i * 4)) & 0xf];
  }
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0';
  
  ebxb[8] = L'\r';
  ebxb[9] = L'\n';
  ebxb[10] = L'\0';
  
  ecxb[8] = L'\r';
  ecxb[9] = L'\n';
  ecxb[10] = L'\0';
  
  edxb[8] = L'\r';
  edxb[9] = L'\n';
  edxb[10] = L'\0';
  
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, ebxb);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, ecxb);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, edxb);
  
  //custom_code();
  
  //UINT32 lowB;
  //UINT32 highB;
  //asm ("rtdsc" : "=a"(lowB), "=d" (highB));
  
  /*
  // potem skasuj - START
													CHAR16 buffer_641[128]; // 64 bit values
												   // buffer initialization
												   
												   _vartest = 0x200; // temp value for tests
												  
												  AsmReadPmc((UINT32)_vartest);

												  for (INTN j = 0; j < 16; j++) {
													buffer_641[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
												  }
												  buffer_641[16] = L'\r'; buffer_641[17] = L'\n'; buffer_641[18] = L'\0'; 
												  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_641);
  // potem skasuj - END
  */
 
  
  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_YELLOW, EFI_GREEN));
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"RTDSC!\r\n\r\n");
  
  
  //AsmInvd(0x0, &eax, &ebx, &ecx, &edx);
  _vartest = 0x1234;
  AsmInvd();
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"read cr0!\r\n\r\n");
  
  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	//ebxb[7 - j] = hexchar[(ebx >> (j * 4)) & 0xf];
	//ecxb[7 - j] = hexchar[(ecx >> (j * 4)) & 0xf];
	//edxb[7 - j] = hexchar[(edx >> (j * 4)) & 0xf];
  }
  //buffer[8] = L'\0';
  
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0'; 
  
  /*
  ebxb[8] = L'\r';
  ebxb[9] = L'\n';
  ebxb[10] = L'\0';
  
  ecxb[8] = L'\r';
  ecxb[9] = L'\n';
  ecxb[10] = L'\0';
  
  edxb[8] = L'\r';
  edxb[9] = L'\n';
  edxb[10] = L'\0';
  */
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"DONE!\r\n\r\n");
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, ebxb);
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, ecxb);
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, edxb);
  
  // 05-01-2025 - 22:04 
  // ok, skoro to dziala już i mogę odtyczyća _vartest jako extern z .nasm
  // to teraz test RDTSC
  
  _vartest = 0;
  
  AsmInvd();

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	//ebxb[7 - j] = hexchar[(ebx >> (j * 4)) & 0xf];
	//ecxb[7 - j] = hexchar[(ecx >> (j * 4)) & 0xf];
	//edxb[7 - j] = hexchar[(edx >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  // demo 2 - ustawiam _vartest = 1
  CHAR16 buffer_var2[64];
  CHAR16 buffer_var3[64];
  _vartest = 1;
  
  // check pointer to reset function
  EFI_RESET_SYSTEM reset_pointer;
  _vartest = (UINT64)&reset_pointer;
  
  AsmInvd();

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	buffer_var2[7 - j] = hexchar[(_vartest2 >> (j * 4)) & 0xf];
	buffer_var3[7 - j] = hexchar[(_vartest3 >> (j * 4)) & 0xf];
	//ebxb[7 - j] = hexchar[(ebx >> (j * 4)) & 0xf];
	//ecxb[7 - j] = hexchar[(ecx >> (j * 4)) & 0xf];
	//edxb[7 - j] = hexchar[(edx >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r';
  buffer[9] = L'\n';
  buffer[10] = L'\0';
  
  buffer_var2[8] = L'\r';
  buffer_var2[9] = L'\n';
  buffer_var2[10] = L'\0';
  
  buffer_var3[8] = L'\r';
  buffer_var3[9] = L'\n';
  buffer_var3[10] = L'\0';
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_var2);
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_var3);
  
  // demo x
  _vartest = (UINT64)SystemTable;
  
  _vartest = 0xff; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  //_vartest = (UINT64)SystemTable;
  
  _vartest = 0xa; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  // lable 3 demo
  _vartest = 0x20; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  
  // lable 4 demo
  _vartest = 0x40; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"\r CR read succsesfully \r\n!");
  
   // lable 5 demo
  _vartest = 0x50; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  // lable 6 demo
  _vartest = 0x60; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  
  SystemTable->ConOut->OutputString(SystemTable->ConOut, u"\r Read CR 4 and CR 0 \r\n!");
  
  // lable 7 demo IA32_MPREF
  _vartest = 0x70; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  // lable 8 demo IA32_APREF
  _vartest = 0x80; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 8; j++) {
	buffer[7 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer[8] = L'\r'; buffer[9] = L'\n'; buffer[10] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
  
  // test 64 bitowych wartosci na printf
  //  CHAR16 buffer[64]; // eax
  //
  //
  // 
  
   CHAR16 buffer_64[128]; // 64 bit values
   CHAR16 buffer_64_b[128]; // 64 bit values
   CHAR16 buffer_64_c[128]; // 64 bit values
   // buffer initialization
   
   /*
   _vartest = 0x90; // temp value for tests
  
  AsmReadPmc((UINT32)_vartest);

  for (INTN j = 0; j < 16; j++) {
	buffer_64[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
  }
  buffer_64[16] = L'\r'; buffer_64[17] = L'\n'; buffer_64[18] = L'\0'; 
  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64);
  */
  
  // next demo
  SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_RED, EFI_BLACK));
  //SystemTable->ConOut->OutputString(SystemTable->ConOut, u"\r\n\DEMO 3!\r\n");
  
  // wait for  key event 
  EFI_INPUT_KEY key;
  while (SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &key) != EFI_SUCCESS) {
	//SystemTable->ConOut->SetAttribute(SystemTable->ConOut, EFI_TEXT_ATTR(EFI_RED, EFI_BLACK));
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 10);
	SystemTable->ConOut->OutputString(SystemTable->ConOut, u"\rDEMO 3!");
	
	for (INTN j = 0; j < 8; j++) {
		buffer[7 - j] = hexchar[(_vartest4_timer1 >> (j * 4)) & 0xf];
	}
	  buffer[8] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer);
		
	// inc timer
	_vartest4_timer1++;
	
	// ------ X1 ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 30, 11);
	
	// print 
	_vartest = 0x90; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64[16] = L'\r'; buffer_64[17] = L'\n'; buffer_64[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64);
	  
	// ------ X2 ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 30, 12);
	
	// print 
	_vartest = 0x100; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64_b[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64_b[16] = L'\r'; buffer_64_b[17] = L'\n'; buffer_64_b[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64_b);
	  
	// ------ X3 ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 30, 13);
	
	// print 
	_vartest = 0x110; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64_c[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64_c[16] = L'\r'; buffer_64_c[17] = L'\n'; buffer_64_c[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64_c);
	  
	  // ------ X4 MSR_RAPL_POWER_UNIT ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 40, 15);
	
	// print 
	_vartest = 0x120; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64_c[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64_c[16] = L'\r'; buffer_64_c[17] = L'\n'; buffer_64_c[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64_c);
	
	// ------ X5 MSR_PLATFORM INFO 0xCE ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 40, 16);
	
	// print 
	_vartest = 0x130; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64_c[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64_c[16] = L'\r'; buffer_64_c[17] = L'\n'; buffer_64_c[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64_c);
	
	// ------ X6 MSR PERF STATUS 0x198 ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 40, 17);
	
	// print 
	_vartest = 0x140; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64_c[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64_c[16] = L'\r'; buffer_64_c[17] = L'\n'; buffer_64_c[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64_c);
	  
	 // ------ X7 IA32 PERF CTL 0x199 ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 40, 18);
	
	// print 
	_vartest = 0x150; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64_c[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64_c[16] = L'\r'; buffer_64_c[17] = L'\n'; buffer_64_c[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64_c);
	
	 // ------ X8 MSR_TURBO_RATIO_LIMIT ------
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 40, 19);
	
	// print 
	_vartest = 0x160; // temp value for tests
  
	  AsmReadPmc((UINT32)_vartest);

	  for (INTN j = 0; j < 16; j++) {
		buffer_64_c[15 - j] = hexchar[(_vartest >> (j * 4)) & 0xf];
	  }
	  buffer_64_c[16] = L'\r'; buffer_64_c[17] = L'\n'; buffer_64_c[18] = L'\0'; 
	  SystemTable->ConOut->OutputString(SystemTable->ConOut, buffer_64_c);
	
	
	
  // ---- end loop -----
  } // koniec petli WAIT FOR EVENT
  
  // shutdown procedure
  SystemTable->RuntimeServices->ResetSystem(EfiResetShutdown, EFI_SUCCESS, 0, NULL);
  
  
  //Print(L"Hi\n");
  
  return EFI_SUCCESS;
  
#endif 

}
