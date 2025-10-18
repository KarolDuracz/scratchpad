/** FixedHelloWorld.c
  Updated HelloWorld example with corrected includes and guarded USB definitions.
  Provides two approaches to list USB VID/PID:
   - high-level via EFI_USB_IO_PROTOCOL (if bus driver enumerated devices)
   - low-level via EFI_USB2_HC_PROTOCOL manual enumeration on a root port
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DebugLib.h>

/* Canonical EDK2 protocol headers (preferred) */
#include <Protocol/Usb2HostController.h>  // EFI_USB2_HC_PROTOCOL
#include <Protocol/UsbIo.h>               // EFI_USB_IO_PROTOCOL (high-level)
#include <IndustryStandard/Usb.h>         // USB constants where available

/* ---------------------------
   Guarded fallback definitions
   (only used if the canonical headers don't define them)
   --------------------------- */

#ifndef EFI_USB_DEVICE_SPEED_DEFINED
#define EFI_USB_DEVICE_SPEED_DEFINED
typedef enum {
  EfiUsbLowSpeed = 0,
  EfiUsbFullSpeed = 1,
  EfiUsbHighSpeed = 2,
  EfiUsbSuperSpeed = 3
} EFI_USB_DEVICE_SPEED;
#endif

#ifndef USB_DEV_TO_HOST
#define USB_DEV_TO_HOST 0x80
#endif
#ifndef USB_HOST_TO_DEV
#define USB_HOST_TO_DEV 0x00
#endif
#ifndef USB_STANDARD
#define USB_STANDARD 0x00
#endif
#ifndef USB_DEVICE
#define USB_DEVICE 0x00
#endif

#ifndef USB_REQ_GET_DESCRIPTOR
#define USB_REQ_GET_DESCRIPTOR 0x06
#endif
#ifndef USB_REQ_SET_ADDRESS
#define USB_REQ_SET_ADDRESS   0x05
#endif
#ifndef USB_REQ_SET_CONFIGURATION
#define USB_REQ_SET_CONFIGURATION 0x09
#endif

#ifndef USB_DESC_TYPE_DEVICE
#define USB_DESC_TYPE_DEVICE  0x01
#endif
#ifndef USB_DESC_TYPE_CONFIG
#define USB_DESC_TYPE_CONFIG  0x02
#endif

/*
#ifndef EFI_USB_DATA_DIRECTION_DEFINED
#define EFI_USB_DATA_DIRECTION_DEFINED
typedef enum {
  EfiUsbDataIn  = 0,
  EfiUsbDataOut = 1,
  EfiUsbNoData  = 2
} EFI_USB_DATA_DIRECTION;
#endif

#ifndef EFI_USB_DEVICE_REQUEST_DEFINED
#define EFI_USB_DEVICE_REQUEST_DEFINED
typedef struct {
  UINT8  RequestType;
  UINT8  Request;
  UINT16 Value;
  UINT16 Index;
  UINT16 Length;
} EFI_USB_DEVICE_REQUEST;
#endif

#ifndef EFI_USB_DEVICE_DESCRIPTOR_DEFINED
#define EFI_USB_DEVICE_DESCRIPTOR_DEFINED
typedef struct {
  UINT8  Length;
  UINT8  DescriptorType;
  UINT16 BcdUSB;
  UINT8  DeviceClass;
  UINT8  DeviceSubClass;
  UINT8  DeviceProtocol;
  UINT8  MaxPacketSize0;
  UINT16 IdVendor;
  UINT16 IdProduct;
  UINT16 BcdDevice;
  UINT8  Manufacturer;
  UINT8  Product;
  UINT8  SerialNumber;
  UINT8  NumConfigurations;
} EFI_USB_DEVICE_DESCRIPTOR;
#endif
*/

/* Some older trees may not expose feature macros; provide conservative defaults */
#ifndef EfiUsbPortPower
#define EfiUsbPortPower   8
#endif
#ifndef EfiUsbPortReset
#define EfiUsbPortReset   4
#endif
#ifndef EfiUsbPortEnable
#define EfiUsbPortEnable  1
#endif

#define DEFAULT_TIMEOUT_MS 2000

/* pick a temporary address for the manual enumeration */
STATIC
UINT8
PickTemporaryAddress(VOID)
{
  return 5; /* simple fixed temporary address; ensure not colliding with other devices */
}

/* High-level: enumerate devices via EFI_USB_IO_PROTOCOL */
STATIC
EFI_STATUS
EnumerateViaUsbIo(VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *HandleBuffer = NULL;
  UINTN HandleCount = 0;

  Print(L"--- Approach A: Locate EFI_USB_IO_PROTOCOL handles (high-level)\n");

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Print(L"No EFI_USB_IO_PROTOCOL device handles found (Status=0x%08x)\n", Status);
    if (HandleBuffer) { gBS->FreePool(HandleBuffer); }
    return EFI_NOT_FOUND;
  }

  Print(L"Found %u USB device handle(s)\n", HandleCount);

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    EFI_USB_DEVICE_DESCRIPTOR DevDesc;
    Status = gBS->HandleProtocol(HandleBuffer[i], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
    if (EFI_ERROR(Status) || UsbIo == NULL) {
      Print(L"  Handle[%u]: Could not open EFI_USB_IO_PROTOCOL (Status=0x%08x)\n", i, Status);
      continue;
    }

    Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
    if (EFI_ERROR(Status)) {
      Print(L"  Handle[%u]: UsbGetDeviceDescriptor failed (0x%08x)\n", i, Status);
      continue;
    }

    Print(L"  Handle[%u]: VID=0x%04x PID=0x%04x Class=0x%02x Subclass=0x%02x Protocol=0x%02x\n",
          i, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass, DevDesc.DeviceSubClass, DevDesc.DeviceProtocol);
  }

  gBS->FreePool(HandleBuffer);
  return EFI_SUCCESS;
}

/* Wrapper around ControlTransfer of the host controller protocol */
STATIC
EFI_STATUS
HcControlTransferWrapper(
  IN EFI_USB2_HC_PROTOCOL *Hc,
  IN UINT8                DeviceAddress,
  IN EFI_USB_DEVICE_SPEED DeviceSpeed,
  IN UINTN                MaxPacketLength,
  IN EFI_USB_DEVICE_REQUEST *Request,
  IN EFI_USB_DATA_DIRECTION TransferDirection,
  IN OUT VOID             *Data,
  IN OUT UINTN            *DataLength,
  IN UINTN                TimeoutMs,
  OUT UINT32              *TransferResult
  )
{
  /* The translator argument is optional (NULL) in our usage */
  return Hc->ControlTransfer(Hc, DeviceAddress, DeviceSpeed, MaxPacketLength, Request,
                             TransferDirection, Data, DataLength, TimeoutMs, NULL, TransferResult);
}

/* Low-level manual enumeration of a single root port (zero-based port index) */
STATIC
EFI_STATUS
ManualEnumeratePort(
  IN EFI_USB2_HC_PROTOCOL *Hc,
  IN UINT8                PortIndex
  )
{
  EFI_STATUS Status;
  EFI_USB_PORT_STATUS PortStatus;
  Print(L"--- Approach B: Manual enumeration on host controller, port index %u\n", PortIndex);

  Status = Hc->GetRootHubPortStatus(Hc, PortIndex, &PortStatus);
  if (EFI_ERROR(Status)) {
    Print(L"GetRootHubPortStatus failed (0x%08x)\n", Status);
    return Status;
  }

  Print(L" PortStatus=0x%08x PortChange=0x%08x\n", PortStatus.PortStatus, PortStatus.PortChangeStatus);

  /* check connection bit (bit0 typically) */
  if ((PortStatus.PortStatus & 0x00000001) == 0) {
    Print(L" No device connected on this port (index %u)\n", PortIndex);
    return EFI_NOT_FOUND;
  }

  /* Power the port if not powered (commonly bit 8 = powered) */
  if ((PortStatus.PortStatus & 0x00000100) == 0) {
    Print(L" Powering port ...\n");
    Hc->SetRootHubPortFeature(Hc, PortIndex, EfiUsbPortPower);
    gBS->Stall(2000 * 1000); /* 2 seconds */
  }

  /* Reset the port */
  Print(L" Resetting port ...\n");
  Hc->SetRootHubPortFeature(Hc, PortIndex, EfiUsbPortReset);

  /* Poll for reset completion (simple loop) */
  {
    UINTN poll;
    for (poll = 0; poll < 50; ++poll) {
      gBS->Stall(100 * 1000); /* 100 ms */
      Status = Hc->GetRootHubPortStatus(Hc, PortIndex, &PortStatus);
      if (EFI_ERROR(Status)) {
        Print(L"  GetRootHubPortStatus after reset failed (0x%08x)\n", Status);
        return Status;
      }
      /* USB reset bit often bit4 (0x10). Wait until it clears. */
      if ((PortStatus.PortStatus & 0x00000010) == 0) {
        break;
      }
    }
    if (poll >= 50) {
      Print(L"  Reset polling timed out (continuing)\n");
    }
  }

  /* Derive device speed from port status bits (best-effort) */
  EFI_USB_DEVICE_SPEED deviceSpeed = EfiUsbFullSpeed;
  if (PortStatus.PortStatus & 0x00000200) { /* example low-speed bit test */
    deviceSpeed = EfiUsbLowSpeed;
  } else if (PortStatus.PortStatus & 0x00000400) { /* example high-speed bit test */
    deviceSpeed = EfiUsbHighSpeed;
  } else {
    deviceSpeed = EfiUsbFullSpeed;
  }
  Print(L" Device speed (inferred) = %u\n", deviceSpeed);

  /* Start standard enumeration sequence on default address 0 */

  UINT8 deviceAddress = 0;
  UINTN packetLen = 8; /* small initial read to learn bMaxPacketSize0 */
  EFI_USB_DEVICE_DESCRIPTOR smallDevDesc;
  EFI_USB_DEVICE_REQUEST request;
  UINT32 transferResult = 0;
  UINTN dataLen;

  /* GET_DESCRIPTOR(Device, 8) */
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_DEVICE << 8) | 0;
  request.Index = 0;
  request.Length = 8;

  Print(L"  GET_DESCRIPTOR(Device,8) ...\n");
  dataLen = 8;
  ZeroMem(&smallDevDesc, sizeof(smallDevDesc));
  Status = HcControlTransferWrapper(Hc, deviceAddress, deviceSpeed, packetLen, &request, EfiUsbDataIn, &smallDevDesc, &dataLen, DEFAULT_TIMEOUT_MS, &transferResult);
  if (EFI_ERROR(Status)) {
    Print(L"   GET_DESCRIPTOR(8) failed (0x%08x) result=0x%08x\n", Status, transferResult);
    return Status;
  }

  /* bMaxPacketSize0 usually at offset 7 (MaxPacketSize0) */
  UINT8 bMaxPacket0 = ((UINT8 *)&smallDevDesc)[7];
  if (bMaxPacket0 == 0) {
    bMaxPacket0 = 8;
  }
  Print(L"   bMaxPacketSize0 = %u\n", bMaxPacket0);

  /* SET_ADDRESS to a chosen temporary address */
  UINT8 newAddr = PickTemporaryAddress();
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_HOST_TO_DEV | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_SET_ADDRESS;
  request.Value = newAddr;
  request.Index = 0;
  request.Length = 0;

  dataLen = 0;
  Print(L"  SET_ADDRESS = %u ...\n", newAddr);
  Status = HcControlTransferWrapper(Hc, 0, deviceSpeed, bMaxPacket0, &request, EfiUsbNoData, NULL, &dataLen, DEFAULT_TIMEOUT_MS, &transferResult);
  if (EFI_ERROR(Status)) {
    Print(L"   SET_ADDRESS failed (0x%08x)\n", Status);
    return Status;
  }

  /* short stall to allow device to adopt new address */
  gBS->Stall(10 * 1000); /* 10 ms */

  /* GET_DESCRIPTOR(Device, 18) at new address */
  EFI_USB_DEVICE_DESCRIPTOR devDescFull;
  ZeroMem(&devDescFull, sizeof(devDescFull));
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_DEVICE << 8) | 0;
  request.Index = 0;
  request.Length = sizeof(EFI_USB_DEVICE_DESCRIPTOR);

  dataLen = sizeof(EFI_USB_DEVICE_DESCRIPTOR);
  Print(L"  GET_DESCRIPTOR(Device,full) at addr %u ...\n", newAddr);
  Status = HcControlTransferWrapper(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbDataIn, &devDescFull, &dataLen, DEFAULT_TIMEOUT_MS, &transferResult);
  if (EFI_ERROR(Status)) {
    Print(L"   GET_DESCRIPTOR(Device,full) failed (0x%08x) result=0x%08x\n", Status, transferResult);
    return Status;
  }

  Print(L"   Device: VID=0x%04x PID=0x%04x Class=0x%02x NumConfigs=%u\n",
        devDescFull.IdVendor, devDescFull.IdProduct, devDescFull.DeviceClass, devDescFull.NumConfigurations);

  /* GET_DESCRIPTOR(Configuration header, 9 bytes) to discover wTotalLength */
  UINT8 cfgHeader[9];
  ZeroMem(cfgHeader, sizeof(cfgHeader));
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_CONFIG << 8) | 0;
  request.Index = 0;
  request.Length = sizeof(cfgHeader);

  dataLen = sizeof(cfgHeader);
  Status = HcControlTransferWrapper(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbDataIn, cfgHeader, &dataLen, DEFAULT_TIMEOUT_MS, &transferResult);
  if (EFI_ERROR(Status)) {
    Print(L"   GET_DESCRIPTOR(Config header) failed (0x%08x)\n", Status);
    return Status;
  }

  UINT16 wTotalLength = (UINT16)cfgHeader[2] | ((UINT16)cfgHeader[3] << 8);
  if (wTotalLength < 9) {
    Print(L"   Bad wTotalLength=%u; using 9\n", wTotalLength);
    wTotalLength = 9;
  }
  Print(L"   Configuration wTotalLength = %u\n", wTotalLength);

  /* allocate and fetch full configuration descriptor */
  VOID *cfgFull = AllocateZeroPool(wTotalLength);
  if (cfgFull == NULL) {
    Print(L"   Failed to allocate %u bytes for config\n", wTotalLength);
    return EFI_OUT_OF_RESOURCES;
  }

  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_CONFIG << 8) | 0;
  request.Index = 0;
  request.Length = wTotalLength;

  dataLen = wTotalLength;
  Status = HcControlTransferWrapper(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbDataIn, cfgFull, &dataLen, DEFAULT_TIMEOUT_MS, &transferResult);
  if (EFI_ERROR(Status)) {
    Print(L"   GET_DESCRIPTOR(Config full) failed (0x%08x)\n", Status);
    FreePool(cfgFull);
    return Status;
  }

  /* parse first configuration's bConfigurationValue (offset 5 in config header) */
  if (dataLen >= 9) {
    UINT8 bConfigurationValue = ((UINT8*)cfgFull)[5];
    Print(L"   First config bConfigurationValue = %u\n", bConfigurationValue);

    ZeroMem(&request, sizeof(request));
    request.RequestType = USB_HOST_TO_DEV | USB_STANDARD | USB_DEVICE;
    request.Request = USB_REQ_SET_CONFIGURATION;
    request.Value = bConfigurationValue;
    request.Index = 0;
    request.Length = 0;

    UINTN statusLen = 0;
    Status = HcControlTransferWrapper(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbNoData, NULL, &statusLen, DEFAULT_TIMEOUT_MS, &transferResult);
    if (EFI_ERROR(Status)) {
      Print(L"   SET_CONFIGURATION failed (0x%08x)\n", Status);
    } else {
      Print(L"   SET_CONFIGURATION OK\n");
    }
  } else {
    Print(L"   Configuration descriptor too small to parse\n");
  }

  FreePool(cfgFull);
  return EFI_SUCCESS;
}

/// IO part 
/*
USB enumeration (safe manual mode)
Found 1 USB device handle(s)
 Handle[0] VID=0x058F PID=0x6387

Found 1 host controllers
 Host[0] MaxSpeed=12 NumPorts=2
 Inspecting root port 1 (index 0) of host 0
ManualEnumeratePortSafe: port index 0
 PortStatus=0x00000503 PortChange=0x00000000
 Resetting port
 deviceSpeed inferred = 2
 bMaxPacketSize0 = 64
 Device found VID=0x058F PID=0x6387 configs=1
 config wTotalLength=32
 bConfigurationValue=1
 SET_CONFIGURATION OK
 Port enabled detected
  Bus driver attached device and created EFI_USB_IO handle -- recommended: use USB_IO now
*/

//
// Drop these into your UEFI app (EDK2). Assumes you included:
//   <Protocol/DevicePath.h>
//   <Library/DevicePathLib.h>
//   <Protocol/UsbIo.h>
//   <Library/UefiBootServicesTableLib.h>
//   <Library/UefiLib.h>
//

/*
  FindUsbIoByVidPid:
  - Polls for a short while to let the bus driver attach, returns a handle if found.
*/
EFI_STATUS
FindUsbIoByVidPid(
  IN  UINT16 Vid,
  IN  UINT16 Pid,
  IN  UINTN  PollRetries,       // e.g. 25
  IN  UINTN  PollIntervalMs,    // e.g. 200
  OUT EFI_HANDLE *FoundHandle   // optional out
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;

  for (UINTN attempt = 0; attempt < PollRetries; ++attempt) {
    if (Handles) { gBS->FreePool(Handles); Handles = NULL; Count = 0; }
    Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &Count, &Handles);
    if (!EFI_ERROR(Status) && Count > 0) {
      for (UINTN i = 0; i < Count; ++i) {
        EFI_USB_IO_PROTOCOL *UsbIo = NULL;
        Status = gBS->HandleProtocol(Handles[i], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
        if (EFI_ERROR(Status) || UsbIo == NULL) {
          continue;
        }
        EFI_USB_DEVICE_DESCRIPTOR Desc;
        Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &Desc);
        if (!EFI_ERROR(Status)) {
          if (Desc.IdVendor == Vid && Desc.IdProduct == Pid) {
            if (FoundHandle) { *FoundHandle = Handles[i]; }
            gBS->FreePool(Handles);
            return EFI_SUCCESS;
          }
        }
      }
    }
    /* wait then poll again */
    gBS->Stall((UINT32)PollIntervalMs * 1000);
  }

  if (Handles) { gBS->FreePool(Handles); }
  return EFI_NOT_FOUND;
}

/*
  GetParentPortFromUsbIoHandle:
  - Given a child handle (that has EFI_DEVICE_PATH_PROTOCOL), find the USB device path node
    and return the ParentPortNumber (1-based). Caller converts to zero-based if needed.
*/
EFI_STATUS
GetParentPortFromUsbIoHandle(
  IN  EFI_HANDLE Handle,
  OUT UINT8     *ParentPortNumber  // 1-based per UEFI spec
  )
{
  if (Handle == NULL || ParentPortNumber == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  EFI_DEVICE_PATH_PROTOCOL *DevPath = DevicePathFromHandle(Handle);
  if (DevPath == NULL) {
    return EFI_NOT_FOUND;
  }

  EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
  while (!IsDevicePathEnd(Node)) {
    UINT8 Type = DevicePathType(Node);
    UINT8 SubType = DevicePathSubType(Node);

    // Messaging / USB Device node subtype is MSG_USB_DP (0x05).
    // In EDK2 headers MSG_USB_DP is defined in DevicePath.h.
    if (Type == MESSAGING_DEVICE_PATH && SubType == MSG_USB_DP) {
      // The USB device path node layout (UEFI spec) is:
      // EFI_DEVICE_PATH_PROTOCOL Header;
      // UINT8 ParentPortNumber;    // 1-based
      // UINT8 InterfaceNumber;
      // So we can cast and access ParentPortNumber.
      typedef struct {
        EFI_DEVICE_PATH_PROTOCOL Header;
        UINT8  ParentPortNumber;
        UINT8  InterfaceNumber;
      } USB_DEVICE_PATH_NODE;

      USB_DEVICE_PATH_NODE *UsbNode = (USB_DEVICE_PATH_NODE *)Node;
      *ParentPortNumber = UsbNode->ParentPortNumber;
      return EFI_SUCCESS;
    }

    Node = NextDevicePathNode(Node);
  }

  return EFI_NOT_FOUND;
}

/*
  DumpUsbIoDescriptors:
  - Uses EFI_USB_IO_PROTOCOL helpers to get device and config descriptors and prints VID/PID.
*/
EFI_STATUS
DumpUsbIoDescriptors(
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;
  EFI_USB_IO_PROTOCOL *UsbIo = NULL;

  Status = gBS->HandleProtocol(Handle, &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
  if (EFI_ERROR(Status) || UsbIo == NULL) {
    Print(L"DumpUsbIoDescriptors: HandleProtocol UsbIo failed (0x%08x)\n", Status);
    return Status;
  }

  EFI_USB_DEVICE_DESCRIPTOR DevDesc;
  Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
  if (EFI_ERROR(Status)) {
    Print(L"UsbGetDeviceDescriptor failed (0x%08x)\n", Status);
    return Status;
  }

  Print(L"UsbIo device: VID=0x%04x PID=0x%04x Class=0x%02x\n", DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);

  // Read the full config descriptor using helper:
  EFI_USB_CONFIG_DESCRIPTOR *CfgDesc = NULL;
  Status = UsbIo->UsbGetConfigDescriptor(UsbIo, 0, NULL, NULL); // some implementations require an allocated buffer; prefer UsbGetConfigDescriptor with buffer pointer
  // Safer: use UsbGetConfigDescriptor with length query pattern:
  UINTN BufferSize = 0;
  Status = UsbIo->UsbGetConfigDescriptor(UsbIo, 0, NULL, &BufferSize);
  if (Status == EFI_BUFFER_TOO_SMALL && BufferSize > 0) {
    CfgDesc = AllocateZeroPool(BufferSize);
    if (CfgDesc) {
      Status = UsbIo->UsbGetConfigDescriptor(UsbIo, 0, CfgDesc, &BufferSize);
      if (!EFI_ERROR(Status)) {
        Print(L"  Config wTotalLength=%u, NumInterfaces (first)=maybe parse from descriptor\n", CfgDesc->TotalLength);
      } else {
        Print(L"  UsbGetConfigDescriptor(full) failed (0x%08x)\n", Status);
      }
      FreePool(CfgDesc);
    }
  } else {
    // If the above pattern isn't supported (some UsbIo implementations provide direct helper)
    // we can fallback to UsbControlTransfer via UsbIo->UsbControlTransfer (but that defeats the benefit
    // of using UsbIo helper functions).
  }

  return EFI_SUCCESS;
}


/* Main entry: combine both approaches and fix includes/definitions */
EFI_STATUS
EFIAPI
UefiMain(
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;

  Print(L"\nUSB Port Enumeration (fixed headers/defines)\n");

  /* Approach A */
  Status = EnumerateViaUsbIo();
  Print(L"\n");


  /* Approach B: find host controllers and run manual enumeration on root port 1 (index 0) */

  EFI_HANDLE *HandleBuffer = NULL;
  UINTN HandleCount = 0;

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Print(L"No EFI_USB2_HC_PROTOCOL handles found (Status=0x%08x)\n", Status);
    return EFI_NOT_FOUND;
  }

  Print(L"Found %u USB host controller handle(s)\n", HandleCount);

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_USB2_HC_PROTOCOL *Hc = NULL;
    Status = gBS->HandleProtocol(HandleBuffer[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Hc);
    if (EFI_ERROR(Status) || Hc == NULL) {
      Print(L"  Handle[%u]: Could not open EFI_USB2_HC_PROTOCOL (0x%08x)\n", i, Status);
      continue;
    }

    UINT8 MaxSpeed = 0;
    UINT8 NumPorts = 0;
    UINT8 Is64 = 0;
    Status = Hc->GetCapability(Hc, &MaxSpeed, &NumPorts, &Is64);
    if (!EFI_ERROR(Status)) {
      Print(L"  Host[%u]: MaxSpeed=%u NumPorts=%u 64bit=%u\n", i, MaxSpeed, NumPorts, Is64);
    } else {
      Print(L"  Host[%u]: GetCapability failed (0x%08x)\n", i, Status);
    }

    if (NumPorts >= 1) {
      UINT8 portIndex = 0; /* human "Port 1" -> index 0 */
      Print(L"  Inspecting host[%u] root port (human 1 -> index 0)\n", i);
      Status = ManualEnumeratePort(Hc, portIndex);
      Print(L"\n");
    } else {
      Print(L"  Host[%u] reports no ports\n", i);
    }
  }

  if (HandleBuffer) { gBS->FreePool(HandleBuffer); }
  return EFI_SUCCESS;
*/  
  
}
