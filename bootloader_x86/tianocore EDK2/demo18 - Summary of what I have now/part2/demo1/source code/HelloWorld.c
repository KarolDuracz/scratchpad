/* UsbPortEnum_FixRace.c
   Updated from previous example — avoids freeze by polling for EFI_USB_IO
   after SET_CONFIGURATION and using timeouts and port-enable checks.
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DebugLib.h>

#include <Protocol/Usb2HostController.h>
#include <Protocol/UsbIo.h>
#include <IndustryStandard/Usb.h>

#include <Library/BaseLib.h>            // CompareGuid

#include <Library/DevicePathLib.h>
#include <Protocol/DevicePath.h>

// GOP
#include <Protocol/GraphicsOutput.h>

static EFI_GRAPHICS_OUTPUT_PROTOCOL *mGraphicsOuput = NULL;

EFI_GRAPHICS_OUTPUT_BLT_PIXEL white = { 255, 255, 255, 0 };
EFI_GRAPHICS_OUTPUT_BLT_PIXEL blue = { 255, 234, 0, 0 };


#ifndef EfiUsbPortPower
#define EfiUsbPortPower   8
#endif
#ifndef EfiUsbPortReset
#define EfiUsbPortReset   4
#endif
#ifndef EfiUsbPortEnable
#define EfiUsbPortEnable  1
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

#define DEFAULT_TIMEOUT_MS 2000
#define USB_IO_POLL_INTERVAL_MS 200
#define USB_IO_POLL_RETRIES 25 /* ~5s poll */

//#define DEFAULT_TIMEOUT_MS     2000
#define IO_POLL_INTERVAL_MS    200
#define IO_POLL_RETRIES        25


#ifndef EFI_USB_DEVICE_SPEED_DEFINED
#define EFI_USB_DEVICE_SPEED_DEFINED
typedef enum {
  EfiUsbLowSpeed = 0,
  EfiUsbFullSpeed = 1,
  EfiUsbHighSpeed = 2,
  EfiUsbSuperSpeed = 3
} EFI_USB_DEVICE_SPEED;
#endif

STATIC
UINT8
PickTemporaryAddress(VOID)
{
  return 5;
}

/* Constants */
#define PCI_BAR0_OFFSET 0x10
#define NUM_PCI_BARS 6
//#define PORT_INDEX_UNKNOWN 0xFF

/* Forward prototypes (declare before use) */
//EFI_STATUS FindPciIoHandleForDevice(IN EFI_HANDLE SourceHandle, OUT EFI_HANDLE *PciHandleOut);
//EFI_STATUS PrintPciBarsFromPciIo(IN EFI_PCI_IO_PROTOCOL *PciIo);
//EFI_STATUS ShowPciDeviceBarsForHandle(IN EFI_HANDLE SourceHandle);



/* Forward declarations of helper functions (implementations follow) */
EFI_STATUS FindUsbIoByVidPid(IN UINT16 Vid, IN UINT16 Pid, IN UINTN PollRetries, IN UINTN PollIntervalMs, OUT EFI_HANDLE *FoundHandle);
EFI_STATUS GetParentPortFromUsbIoHandle(IN EFI_HANDLE Handle, OUT UINT8 *ParentPortNumber);
EFI_STATUS DumpUsbIoDescriptors(IN EFI_HANDLE Handle);
EFI_STATUS EnumerateViaUsbIo(VOID);

/* Utility: search existing UsbIo handles for a device by VID/PID.
   Polls a few times until timeout to allow bus driver to create handle. */
/*
STATIC
EFI_STATUS
FindUsbIoByVidPid(
  IN UINT16 Vid,
  IN UINT16 Pid,
  IN UINTN  PollRetries,
  IN UINTN  PollIntervalMs,
  OUT EFI_HANDLE *FoundHandle  // optional out 
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
    // wait then poll again
    gBS->Stall((UINT32)PollIntervalMs * 1000);
  }

  if (Handles) { gBS->FreePool(Handles); }
  return EFI_NOT_FOUND;
}
*/

/* Wrapper for ControlTransfer with translator NULL */
STATIC
EFI_STATUS
HcControlTransfer(
  IN EFI_USB2_HC_PROTOCOL *Hc,
  IN UINT8 DeviceAddress,
  IN EFI_USB_DEVICE_SPEED DeviceSpeed,
  IN UINTN MaxPacketLength,
  IN EFI_USB_DEVICE_REQUEST *Request,
  IN EFI_USB_DATA_DIRECTION Direction,
  IN OUT VOID *Data,
  IN OUT UINTN *DataLength,
  IN UINTN TimeoutMs,
  OUT UINT32 *TransferResult
  )
{
  return Hc->ControlTransfer(Hc, DeviceAddress, DeviceSpeed, MaxPacketLength, Request,
                             Direction, Data, DataLength, TimeoutMs, NULL, TransferResult);
}

/* Manual enumeration with safety: after SET_CONFIGURATION we poll for USB_IO handle and bail safely. */
STATIC
EFI_STATUS
ManualEnumeratePortSafe(
  IN EFI_USB2_HC_PROTOCOL *Hc,
  IN UINT8 PortIndex
  )
{
  EFI_STATUS Status;
  EFI_USB_PORT_STATUS PortStatus;
  Print(L"ManualEnumeratePortSafe: port index %u\n", PortIndex);

  Status = Hc->GetRootHubPortStatus(Hc, PortIndex, &PortStatus);
  if (EFI_ERROR(Status)) {
    Print(L" GetRootHubPortStatus failed 0x%08x\n", Status);
    return Status;
  }

  Print(L" PortStatus=0x%08x PortChange=0x%08x\n", PortStatus.PortStatus, PortStatus.PortChangeStatus);

  if ((PortStatus.PortStatus & 0x1) == 0) {
    Print(L" No device present\n");
    return EFI_NOT_FOUND;
  }

  /* power if needed */
  if ((PortStatus.PortStatus & 0x00000100) == 0) {
    Print(L" Powering port\n");
    Hc->SetRootHubPortFeature(Hc, PortIndex, EfiUsbPortPower);
    gBS->Stall(500 * 1000); /* 500 ms */
  }

  /* Reset the port */
  Print(L" Resetting port\n");
  Hc->SetRootHubPortFeature(Hc, PortIndex, EfiUsbPortReset);

  /* Poll for reset complete */
  {
    UINTN tries = 0;
    for (; tries < 30; ++tries) {
      gBS->Stall(100 * 1000);
      Status = Hc->GetRootHubPortStatus(Hc, PortIndex, &PortStatus);
      if (EFI_ERROR(Status)) return Status;
      if ((PortStatus.PortStatus & 0x10) == 0) break; /* reset cleared */
    }
    if (tries >= 30) {
      Print(L" Reset may have timed out (continuing)\n");
    }
  }

  /* infer speed */
  EFI_USB_DEVICE_SPEED deviceSpeed = EfiUsbFullSpeed;
  if (PortStatus.PortStatus & 0x00000400) deviceSpeed = EfiUsbHighSpeed;
  else if (PortStatus.PortStatus & 0x00000200) deviceSpeed = EfiUsbLowSpeed;
  Print(L" deviceSpeed inferred = %u\n", deviceSpeed);

  UINT8 defaultAddr = 0;
  UINTN packetLen = 8;
  EFI_USB_DEVICE_DESCRIPTOR smallDevDesc;
  EFI_USB_DEVICE_REQUEST request;
  UINT32 xferResult = 0;
  UINTN dataLen = 0;

  /* GET_DESCRIPTOR(Device, 8) */
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_DEVICE << 8) | 0;
  request.Index = 0;
  request.Length = 8;
  dataLen = 8;
  ZeroMem(&smallDevDesc, sizeof(smallDevDesc));
  Status = HcControlTransfer(Hc, defaultAddr, deviceSpeed, packetLen, &request, EfiUsbDataIn, &smallDevDesc, &dataLen, DEFAULT_TIMEOUT_MS, &xferResult);
  if (EFI_ERROR(Status)) {
    Print(L" GET_DESCRIPTOR(8) failed 0x%08x result=0x%08x\n", Status, xferResult);
    return Status;
  }
  UINT8 bMaxPacket0 = ((UINT8*)&smallDevDesc)[7];
  if (bMaxPacket0 == 0) bMaxPacket0 = 8;
  Print(L" bMaxPacketSize0 = %u\n", bMaxPacket0);

  /* SET_ADDRESS */
  UINT8 newAddr = PickTemporaryAddress();
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_HOST_TO_DEV | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_SET_ADDRESS;
  request.Value = newAddr;
  request.Index = 0;
  request.Length = 0;
  dataLen = 0;
  Status = HcControlTransfer(Hc, 0, deviceSpeed, bMaxPacket0, &request, EfiUsbNoData, NULL, &dataLen, DEFAULT_TIMEOUT_MS, &xferResult);
  if (EFI_ERROR(Status)) {
    Print(L" SET_ADDRESS failed 0x%08x\n", Status);
    return Status;
  }
  gBS->Stall(10 * 1000); /* 10 ms */

  /* GET_DESCRIPTOR(Device, full) */
  EFI_USB_DEVICE_DESCRIPTOR devFull;
  ZeroMem(&devFull, sizeof(devFull));
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_DEVICE << 8) | 0;
  request.Index = 0;
  request.Length = sizeof(EFI_USB_DEVICE_DESCRIPTOR);
  dataLen = sizeof(EFI_USB_DEVICE_DESCRIPTOR);
  Status = HcControlTransfer(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbDataIn, &devFull, &dataLen, DEFAULT_TIMEOUT_MS, &xferResult);
  if (EFI_ERROR(Status)) {
    Print(L" GET_DESCRIPTOR(full) failed 0x%08x result=0x%08x\n", Status, xferResult);
    return Status;
  }
  Print(L" Device found VID=0x%04x PID=0x%04x configs=%u\n", devFull.IdVendor, devFull.IdProduct, devFull.NumConfigurations);

  /* GET_CONFIG header (9) */
  UINT8 cfgHdr[9];
  ZeroMem(cfgHdr, sizeof(cfgHdr));
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_CONFIG << 8) | 0;
  request.Index = 0;
  request.Length = sizeof(cfgHdr);
  dataLen = sizeof(cfgHdr);
  Status = HcControlTransfer(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbDataIn, cfgHdr, &dataLen, DEFAULT_TIMEOUT_MS, &xferResult);
  if (EFI_ERROR(Status)) {
    Print(L" GET_DESCRIPTOR(config header) failed 0x%08x\n", Status);
    return Status;
  }
  UINT16 wTotalLength = (UINT16)cfgHdr[2] | ((UINT16)cfgHdr[3] << 8);
  if (wTotalLength < 9) wTotalLength = 9;
  Print(L" config wTotalLength=%u\n", wTotalLength);

  VOID *cfgBuf = AllocateZeroPool(wTotalLength);
  if (!cfgBuf) return EFI_OUT_OF_RESOURCES;
  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_GET_DESCRIPTOR;
  request.Value = (USB_DESC_TYPE_CONFIG << 8) | 0;
  request.Index = 0;
  request.Length = wTotalLength;
  dataLen = wTotalLength;
  Status = HcControlTransfer(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbDataIn, cfgBuf, &dataLen, DEFAULT_TIMEOUT_MS, &xferResult);
  if (EFI_ERROR(Status)) {
    Print(L" GET_DESCRIPTOR(config full) failed 0x%08x\n", Status);
    FreePool(cfgBuf);
    return Status;
  }

  /* set configuration (first config value at offset 5 of config header) */
  UINT8 bConfigurationValue = ((UINT8*)cfgBuf)[5];
  Print(L" bConfigurationValue=%u\n", bConfigurationValue);

  ZeroMem(&request, sizeof(request));
  request.RequestType = USB_HOST_TO_DEV | USB_STANDARD | USB_DEVICE;
  request.Request = USB_REQ_SET_CONFIGURATION;
  request.Value = bConfigurationValue;
  request.Index = 0;
  request.Length = 0;
  dataLen = 0;
  Status = HcControlTransfer(Hc, newAddr, deviceSpeed, bMaxPacket0, &request, EfiUsbNoData, NULL, &dataLen, DEFAULT_TIMEOUT_MS, &xferResult);
  if (EFI_ERROR(Status)) {
    Print(L" SET_CONFIGURATION failed 0x%08x\n", Status);
    FreePool(cfgBuf);
    return Status;
  }
  Print(L" SET_CONFIGURATION OK\n");

  /* --- NEW: after SET_CONFIGURATION, wait briefly then check port enabled and poll for bus-driver-created EFI_USB_IO handle --- */

  /* small settle time */
  gBS->Stall(50 * 1000); /* 50 ms */

  /* Re-read port status; ensure 'Enabled' bit is set (usually bit1). If not, poll a bit. */
  {
    UINTN tries = 0;
    for (; tries < 20; ++tries) {
      Status = Hc->GetRootHubPortStatus(Hc, PortIndex, &PortStatus);
      if (EFI_ERROR(Status)) break;
      if (PortStatus.PortStatus & 0x02) break; /* Enabled bit often bit1 */
      gBS->Stall(50 * 1000);
    }
    if (tries >= 20) {
      Print(L" Warning: port not reporting Enabled after SET_CONFIGURATION (continuing)\n");
    } else {
      Print(L" Port enabled detected\n");
    }
  }

  /* Poll for EFI_USB_IO handle for this VID/PID for a few seconds to allow bus driver to attach */
  EFI_STATUS found = FindUsbIoByVidPid(devFull.IdVendor, devFull.IdProduct, USB_IO_POLL_RETRIES, USB_IO_POLL_INTERVAL_MS, NULL);
  if (!EFI_ERROR(found)) {
    Print(L"  Bus driver attached device and created EFI_USB_IO handle -- recommended: use USB_IO now\n");
    FreePool(cfgBuf);
    return EFI_SUCCESS;
  } else {
    Print(L"  No EFI_USB_IO handle found for VID/PID after wait; continuing with HC for now\n");
  }

  /* If we continue, be careful and do not issue more lengthy transfers -- for demo we stop here */
  FreePool(cfgBuf);
  return EFI_SUCCESS;
}



/* --------------------------
   Implementations
   -------------------------- */

/* FindUsbIoByVidPid
   Poll for a short while for the bus driver to attach and create an EFI_USB_IO handle
   matching the provided VID/PID. If FoundHandle != NULL, it is filled with the handle.
*/
EFI_STATUS
FindUsbIoByVidPid(
  IN UINT16 Vid,
  IN UINT16 Pid,
  IN UINTN  PollRetries,
  IN UINTN  PollIntervalMs,
  OUT EFI_HANDLE *FoundHandle
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
            if (FoundHandle) *FoundHandle = Handles[i];
            gBS->FreePool(Handles);
            return EFI_SUCCESS;
          }
        }
      }
    }

    /* sleep between polls */
    gBS->Stall((UINT32)PollIntervalMs * 1000);
  }

  if (Handles) gBS->FreePool(Handles);
  return EFI_NOT_FOUND;
}

/* GetParentPortFromUsbIoHandle
   Walk the device path of the supplied handle and return the ParentPortNumber
   from the first USB device-path node found (1-based).
*/
EFI_STATUS
GetParentPortFromUsbIoHandle(
  IN EFI_HANDLE Handle,
  OUT UINT8 *ParentPortNumber
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
    if (DevicePathType(Node) == MESSAGING_DEVICE_PATH && DevicePathSubType(Node) == MSG_USB_DP) {
      /* USB device path node layout:
         EFI_DEVICE_PATH_PROTOCOL Header;
         UINT8 ParentPortNumber;    // 1-based
         UINT8 InterfaceNumber;
      */
      typedef struct {
        EFI_DEVICE_PATH_PROTOCOL Header;
        UINT8 ParentPortNumber;
        UINT8 InterfaceNumber;
      } USB_DEVICE_PATH_NODE;

      USB_DEVICE_PATH_NODE *UsbNode = (USB_DEVICE_PATH_NODE *)Node;
      *ParentPortNumber = UsbNode->ParentPortNumber;
      return EFI_SUCCESS;
    }
    Node = NextDevicePathNode(Node);
  }

  return EFI_NOT_FOUND;
}

/* DumpUsbIoDescriptors
   Uses EFI_USB_IO_PROTOCOL to get the device descriptor, then attempts to fetch the
   full configuration descriptor using the common "query size -> allocate -> fetch" pattern.
*/
EFI_STATUS
DumpUsbIoDescriptors(
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;
  EFI_USB_IO_PROTOCOL *UsbIo = NULL;

  Status = gBS->HandleProtocol(Handle, &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
  if (EFI_ERROR(Status) || UsbIo == NULL) {
    Print(L"DumpUsbIoDescriptors: HandleProtocol UsbIo failed (0x%08x)\n", Status);
    return Status;
  }

  /* Device descriptor */
  EFI_USB_DEVICE_DESCRIPTOR DevDesc;
  Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
  if (EFI_ERROR(Status)) {
    Print(L"  UsbGetDeviceDescriptor failed (0x%08x)\n", Status);
    return Status;
  }

  Print(L"  Device descriptor: VID=0x%04x PID=0x%04x Class=0x%02x NumConfigs=%u\n",
        DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass, DevDesc.NumConfigurations);

  /* Try the UsbGetConfigDescriptor helper (two-parameter form in your tree) */
  EFI_USB_CONFIG_DESCRIPTOR CfgHeader;
  ZeroMem(&CfgHeader, sizeof(CfgHeader));

  Status = UsbIo->UsbGetConfigDescriptor(UsbIo, &CfgHeader);
  if (!EFI_ERROR(Status)) {
    UINT16 totalLen = CfgHeader.TotalLength;
    if (totalLen < sizeof(EFI_USB_CONFIG_DESCRIPTOR)) {
      totalLen = sizeof(EFI_USB_CONFIG_DESCRIPTOR);
    }
    Print(L"  UsbGetConfigDescriptor returned header: TotalLength=%u\n", totalLen);

    VOID *CfgBuf = AllocateZeroPool(totalLen);
    if (CfgBuf == NULL) {
      Print(L"  Failed to allocate %u bytes for full config descriptor\n", totalLen);
      return EFI_OUT_OF_RESOURCES;
    }

    /* Build GET_DESCRIPTOR (CONFIG) request and fetch full config via UsbControlTransfer */
    EFI_USB_DEVICE_REQUEST Req;
    ZeroMem(&Req, sizeof(Req));
    Req.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
    Req.Request     = USB_REQ_GET_DESCRIPTOR;
    Req.Value       = (USB_DESC_TYPE_CONFIG << 8) | 0;
    Req.Index       = 0;
    Req.Length      = totalLen;

    UINT32 transferStatus = 0;
    /* Signature: UsbControlTransfer(This, Request, Direction, Timeout, Data, DataLength, &Status) */
    Status = UsbIo->UsbControlTransfer(UsbIo, &Req, EfiUsbDataIn, (UINT32)DEFAULT_TIMEOUT_MS, CfgBuf, (UINTN)totalLen, &transferStatus);
    if (EFI_ERROR(Status)) {
      Print(L"  UsbControlTransfer(GET_DESCRIPTOR config) failed (0x%08x) xferStatus=0x%08x\n", Status, transferStatus);
      FreePool(CfgBuf);
      return Status;
    }

    EFI_USB_CONFIG_DESCRIPTOR *FullCfg = (EFI_USB_CONFIG_DESCRIPTOR *)CfgBuf;
    Print(L"  Full config read: TotalLength=%u NumInterfaces=%u ConfigurationValue=%u\n",
          FullCfg->TotalLength, FullCfg->NumInterfaces, FullCfg->ConfigurationValue);

    /* Parse interfaces/endpoints from the buffer here if needed */

    FreePool(CfgBuf);
    return EFI_SUCCESS;
  }

  /* If UsbGetConfigDescriptor failed, fallback to control-transfer approach:
     1) GET_DESCRIPTOR(config,9) -> header (to learn wTotalLength)
     2) GET_DESCRIPTOR(config, wTotalLength) -> full buffer
  */
  Print(L"  UsbGetConfigDescriptor failed (0x%08x) - falling back to control-transfer\n", Status);

  EFI_USB_DEVICE_REQUEST ReqHdr;
  ZeroMem(&ReqHdr, sizeof(ReqHdr));
  ReqHdr.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  ReqHdr.Request     = USB_REQ_GET_DESCRIPTOR;
  ReqHdr.Value       = (USB_DESC_TYPE_CONFIG << 8) | 0;
  ReqHdr.Index       = 0;
  ReqHdr.Length      = sizeof(EFI_USB_CONFIG_DESCRIPTOR); /* typically 9 */

  UINT8 HeaderBuf[sizeof(EFI_USB_CONFIG_DESCRIPTOR)];
  UINT32 hdrXferStatus = 0;
  /* Note: Data pointer goes before DataLength in your UsbControlTransfer signature */
  Status = UsbIo->UsbControlTransfer(UsbIo, &ReqHdr, EfiUsbDataIn, (UINT32)DEFAULT_TIMEOUT_MS, HeaderBuf, (UINTN)sizeof(HeaderBuf), &hdrXferStatus);
  if (EFI_ERROR(Status)) {
    Print(L"  Fallback GET_DESCRIPTOR(config header) failed (0x%08x) xferStatus=0x%08x\n", Status, hdrXferStatus);
    return Status;
  }

  UINT16 wTotalLen = (UINT16)HeaderBuf[2] | ((UINT16)HeaderBuf[3] << 8);
  if (wTotalLen < sizeof(EFI_USB_CONFIG_DESCRIPTOR)) {
    wTotalLen = sizeof(EFI_USB_CONFIG_DESCRIPTOR);
  }
  Print(L"  Fallback header reports TotalLength=%u\n", wTotalLen);

  VOID *FullBuf = AllocateZeroPool(wTotalLen);
  if (FullBuf == NULL) {
    Print(L"  Failed to allocate %u bytes for fallback full config\n", wTotalLen);
    return EFI_OUT_OF_RESOURCES;
  }

  EFI_USB_DEVICE_REQUEST ReqFull;
  ZeroMem(&ReqFull, sizeof(ReqFull));
  ReqFull.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
  ReqFull.Request     = USB_REQ_GET_DESCRIPTOR;
  ReqFull.Value       = (USB_DESC_TYPE_CONFIG << 8) | 0;
  ReqFull.Index       = 0;
  ReqFull.Length      = wTotalLen;

  UINT32 fullXferStatus = 0;
  Status = UsbIo->UsbControlTransfer(UsbIo, &ReqFull, EfiUsbDataIn, (UINT32)DEFAULT_TIMEOUT_MS, FullBuf, (UINTN)wTotalLen, &fullXferStatus);
  if (EFI_ERROR(Status)) {
    Print(L"  Fallback full GET_DESCRIPTOR failed (0x%08x) xferStatus=0x%08x\n", Status, fullXferStatus);
    FreePool(FullBuf);
    return Status;
  }

  EFI_USB_CONFIG_DESCRIPTOR *Cfg = (EFI_USB_CONFIG_DESCRIPTOR *)FullBuf;
  Print(L"  Fallback full config read: TotalLength=%u NumInterfaces=%u ConfigurationValue=%u\n",
        Cfg->TotalLength, Cfg->NumInterfaces, Cfg->ConfigurationValue);

  FreePool(FullBuf);
  return EFI_SUCCESS;
}



/* EnumerateViaUsbIo: list existing UsbIo handles (simple helper) */
/*
EFI_STATUS
EnumerateViaUsbIo(VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &Count, &Handles);
  if (EFI_ERROR(Status) || Count == 0) {
    Print(L"No EFI_USB_IO handles found (Status=0x%08x)\n", Status);
    if (Handles) gBS->FreePool(Handles);
    return EFI_NOT_FOUND;
  }

  Print(L"Found %u USB device handles:\n", Count);
  for (UINTN i = 0; i < Count; ++i) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(Handles[i], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
    if (EFI_ERROR(Status) || UsbIo == NULL) continue;

    EFI_USB_DEVICE_DESCRIPTOR Desc;
    Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &Desc);
    if (EFI_ERROR(Status)) continue;

    Print(L" Handle[%u]: VID=0x%04x PID=0x%04x\n", i, Desc.IdVendor, Desc.IdProduct);
  }

  gBS->FreePool(Handles);
  return EFI_SUCCESS;
}
*/

///////////////////////////////////////////////////// 
/*
//******************************************************
// EFI_LOCATE_SEARCH_TYPE
//******************************************************
typedef enum {
   AllHandles,
   ByRegisterNotify,
   ByProtocol
  } EFI_LOCATE_SEARCH_TYPE;
*/

/* EnumerateViaUsbIo - revised to scan AllHandles and inspect protocols on every handle.
   Drop-in replacement for the previous ByProtocol-only version. */
/* EnumerateViaUsbIo - revised to scan AllHandles and inspect protocols on every handle.
   Fixed versions: variable names, ProtocolsPerHandle usage, ConvertDevicePathToText usage. */
EFI_STATUS
EnumerateViaUsbIo(VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *HandleArray = NULL;    /* array of handles returned by LocateHandleBuffer */
  UINTN HandleCount = 0;

  /* NOTE: first param is the enum AllHandles (EFI_LOCATE_SEARCH_TYPE), not the variable. */
  Status = gBS->LocateHandleBuffer(AllHandles, NULL, NULL, &HandleCount, &HandleArray);
  if (EFI_ERROR(Status) || HandleCount == 0) {
    Print(L"LocateHandleBuffer(AllHandles) failed (0x%08x) or returned zero handles\n", Status);
    if (HandleArray) FreePool(HandleArray);
    return EFI_NOT_FOUND;
  }

  Print(L"Enumerating all %u handles in handle database\n", HandleCount);

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_GUID **ProtocolBuffer = NULL; /* note: double-star */
    UINTN ProtocolCount = 0;
    BOOLEAN hasUsbIo = FALSE;
    BOOLEAN hasUsbHc = FALSE;

    Print(L"\nHandle[%u] = 0x%p\n", (UINT32)i, HandleArray[i]);

    /* Print device path (if present) using ConvertDevicePathToText */
    {
      EFI_DEVICE_PATH_PROTOCOL *DevPath = DevicePathFromHandle(HandleArray[i]);
      if (DevPath) {
        CHAR16 *DpStr = ConvertDevicePathToText(DevPath, TRUE, TRUE);
        if (DpStr) {
          Print(L" DevicePath: %s\n", DpStr);
          FreePool(DpStr);
        }
      }
    }

    /* Get list of protocols installed on this handle.
       ProtocolsPerHandle expects EFI_GUID *** (pass address of EFI_GUID ** variable) */
    Status = gBS->ProtocolsPerHandle(HandleArray[i], &ProtocolBuffer, &ProtocolCount);
    if (EFI_ERROR(Status)) {
      Print(L"  ProtocolsPerHandle failed (0x%08x)\n", Status);
      continue;
    }

    Print(L"  Protocols installed: %u\n", ProtocolCount);
    for (UINTN p = 0; p < ProtocolCount; ++p) {
      EFI_GUID *Pg = ProtocolBuffer[p]; /* each element is an EFI_GUID* */
      if (CompareGuid(Pg, &gEfiUsbIoProtocolGuid)) {
        Print(L"   - EFI_USB_IO_PROTOCOL\n");
        hasUsbIo = TRUE;
      } else if (CompareGuid(Pg, &gEfiUsb2HcProtocolGuid)) {
        Print(L"   - EFI_USB2_HC_PROTOCOL\n");
        hasUsbHc = TRUE;
      } else if (CompareGuid(Pg, &gEfiDevicePathProtocolGuid)) {
        Print(L"   - EFI_DEVICE_PATH_PROTOCOL\n");
      } else {
        /* Print GUID head for debugging */
        Print(L"   - Other protocol: %08x-%04x-%04x...\n",
              Pg->Data1, Pg->Data2, Pg->Data3);
      }
    }

    /* If this handle implements USB_IO, dump descriptors with your helper */
    if (hasUsbIo) {
      Print(L"  --> Detected EFI_USB_IO on this handle; dumping descriptors\n");
      DumpUsbIoDescriptors(HandleArray[i]);
    }

    /* If this handle implements USB2_HC, query capability and list root-hub ports */
    if (hasUsbHc) {
      EFI_USB2_HC_PROTOCOL *Hc = NULL;
      Status = gBS->HandleProtocol(HandleArray[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Hc);
      if (!EFI_ERROR(Status) && Hc != NULL) {
        UINT8 MaxSpeed = 0, NumPorts = 0, Is64 = 0;
        Status = Hc->GetCapability(Hc, &MaxSpeed, &NumPorts, &Is64);
        if (EFI_ERROR(Status)) {
          Print(L"  GetCapability failed (0x%08x)\n", Status);
        } else {
          Print(L"  Host Controller: MaxSpeed=%u NumPorts=%u Is64=%u\n", MaxSpeed, NumPorts, Is64);

          for (UINT8 port = 1; port <= NumPorts; ++port) {
            EFI_USB_PORT_STATUS PortStatus;
            Status = Hc->GetRootHubPortStatus(Hc, port, &PortStatus);
            if (EFI_ERROR(Status)) {
              Print(L"   Port %u: GetRootHubPortStatus failed 0x%08x\n", port, Status);
              continue;
            }

            BOOLEAN present = (PortStatus.PortStatus & 0x01) != 0;
            Print(L"   Port %u: Present=%u Status=0x%08x Change=0x%08x\n",
                  port, present ? 1 : 0, PortStatus.PortStatus, PortStatus.PortChangeStatus);

            /* Optionally call ManualEnumeratePortSafe(Hc, port) for present ports (risky) */
          }
        }
      } else {
        Print(L"  HandleProtocol(EFI_USB2_HC_PROTOCOL) failed (0x%08x)\n", Status);
      }
    }

    if (ProtocolBuffer) {
      FreePool(ProtocolBuffer);
      ProtocolBuffer = NULL;
    }
  }

  if (HandleArray) FreePool(HandleArray);
  return EFI_SUCCESS;
}

typedef struct {
  EFI_DEVICE_PATH_PROTOCOL Header;
  UINT8 ParentPortNumber;    // 1-based
  UINT8 InterfaceNumber;
} USB_DEVICE_PATH_NODE;

/* Compare if 'Prefix' device path is a prefix of 'Full'. If it is, set *NextNodeOut to point
   to the node in Full that immediately follows Prefix. */
#if 0
STATIC
BOOLEAN
IsDevicePathPrefix(
  IN EFI_DEVICE_PATH_PROTOCOL *Prefix,
  IN EFI_DEVICE_PATH_PROTOCOL *Full,
  OUT EFI_DEVICE_PATH_PROTOCOL **NextNodeOut
  )
{
  EFI_DEVICE_PATH_PROTOCOL *P = Prefix;
  EFI_DEVICE_PATH_PROTOCOL *F = Full;

  while (!IsDevicePathEnd(P) && !IsDevicePathEnd(F)) {
    //UINT16 Plen = DevicePathNodeLength(P);
    //UINT16 Flen = DevicePathNodeLength(F);
	UINTN Plen = DevicePathNodeLength(P);
    UINTN Flen = DevicePathNodeLength(F);
    if (Plen != Flen) {
      return FALSE;
    }
    if (CompareMem(P, F, Plen) != 0) {
      return FALSE;
    }
    P = NextDevicePathNode(P);
    F = NextDevicePathNode(F);
  }

  /* If we ended because Prefix ended (i.e. matched all its nodes), success. */
  if (IsDevicePathEnd(P)) {
    /* F now points to the node immediately after the prefix in Full */
    if (NextNodeOut) *NextNodeOut = F;
    return TRUE;
  }

  return FALSE;
}
#endif

/* FindHostControllerAndPortForUsbIo
   For a given EFI_USB_IO handle, attempts to discover the host-controller handle and the
   parent root-hub port index (zero-based). On success, returns EFI_SUCCESS and fills
   *HcHandleOut (optional) and *PortIndexOut (optional). */
#if 0
EFI_STATUS
FindHostControllerAndPortForUsbIo(
  IN EFI_HANDLE UsbIoHandle,
  OUT EFI_HANDLE *HcHandleOut,     /* optional */
  OUT UINT8 *PortIndexOut          /* optional, zero-based */
  )
{
  EFI_STATUS Status;
  EFI_DEVICE_PATH_PROTOCOL *DevPath = DevicePathFromHandle(UsbIoHandle);
  if (DevPath == NULL) {
    return EFI_NOT_FOUND;
  }

  /* First, try to find the first USB device-path node and use its ParentPortNumber if set */
  EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
  while (!IsDevicePathEnd(Node)) {
    if (DevicePathType(Node) == MESSAGING_DEVICE_PATH && DevicePathSubType(Node) == MSG_USB_DP) {
      USB_DEVICE_PATH_NODE *UsbNode = (USB_DEVICE_PATH_NODE *)Node;
      if (UsbNode->ParentPortNumber != 0) {
        if (PortIndexOut) *PortIndexOut = (UINT8)(UsbNode->ParentPortNumber - 1);
        if (HcHandleOut) *HcHandleOut = NULL; /* we don't know the HC handle here */
        return EFI_SUCCESS;
      }
      break; /* found USB node but ParentPortNumber == 0, fallthrough to HC-prefix method */
    }
    Node = NextDevicePathNode(Node);
  }

  /* If ParentPortNumber was 0 or not present, enumerate USB host controllers and try prefix matching */
  EFI_HANDLE *HcHandles = NULL;
  UINTN HcCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HcCount, &HcHandles);
  if (EFI_ERROR(Status) || HcCount == 0) {
    if (HcHandles) gBS->FreePool(HcHandles);
    return EFI_NOT_FOUND;
  }

  for (UINTN h = 0; h < HcCount; ++h) {
    EFI_DEVICE_PATH_PROTOCOL *HcPath = DevicePathFromHandle(HcHandles[h]);
    if (HcPath == NULL) continue;

    EFI_DEVICE_PATH_PROTOCOL *NextNode = NULL;
    if (IsDevicePathPrefix(HcPath, DevPath, &NextNode)) {
      /* NextNode should be the USB node for this device */
      if (NextNode && !IsDevicePathEnd(NextNode) &&
          DevicePathType(NextNode) == MESSAGING_DEVICE_PATH &&
          DevicePathSubType(NextNode) == MSG_USB_DP) {
        USB_DEVICE_PATH_NODE *UsbNode = (USB_DEVICE_PATH_NODE *)NextNode;
        if (UsbNode->ParentPortNumber != 0) {
          if (PortIndexOut) *PortIndexOut = (UINT8)(UsbNode->ParentPortNumber - 1);
          if (HcHandleOut) *HcHandleOut = HcHandles[h];
          gBS->FreePool(HcHandles);
          return EFI_SUCCESS;
        } else {
          /* USB node present but ParentPortNumber==0 — still considered not found */
          if (HcHandleOut) *HcHandleOut = HcHandles[h];
          gBS->FreePool(HcHandles);
          return EFI_NOT_FOUND;
        }
      } else {
        /* Matching prefix but next node not USB (peculiar) */
        if (HcHandleOut) *HcHandleOut = HcHandles[h];
        gBS->FreePool(HcHandles);
        return EFI_NOT_FOUND;
      }
    }
  }

  /* not found */
  if (HcHandles) gBS->FreePool(HcHandles);
  return EFI_NOT_FOUND;
}
#endif

#define PORT_INDEX_UNKNOWN 0xFF

/*
typedef struct {
  EFI_DEVICE_PATH_PROTOCOL Header;
  UINT8 ParentPortNumber;    // 1-based
  UINT8 InterfaceNumber;
} USB_DEVICE_PATH_NODE;
*/

/* Return TRUE if Prefix is a prefix of Full. NextNodeOut points to the node in Full immediately after Prefix. */
STATIC
BOOLEAN
IsDevicePathPrefix(
  IN EFI_DEVICE_PATH_PROTOCOL *Prefix,
  IN EFI_DEVICE_PATH_PROTOCOL *Full,
  OUT EFI_DEVICE_PATH_PROTOCOL **NextNodeOut
  )
{
  EFI_DEVICE_PATH_PROTOCOL *P = Prefix;
  EFI_DEVICE_PATH_PROTOCOL *F = Full;

  while (!IsDevicePathEnd(P) && !IsDevicePathEnd(F)) {
    UINTN Plen = DevicePathNodeLength(P);
    UINTN Flen = DevicePathNodeLength(F);
    if (Plen != Flen) return FALSE;
    if (CompareMem(P, F, Plen) != 0) return FALSE;
    P = NextDevicePathNode(P);
    F = NextDevicePathNode(F);
  }

  if (IsDevicePathEnd(P)) {
    if (NextNodeOut) *NextNodeOut = F;
    return TRUE;
  }

  return FALSE;
}

/* FindHostControllerAndPortForUsbIo:
   - UsbIoHandle: handle implementing EFI_USB_IO (child)
   - HcHandleOut: optional out, will be set to the matching HC handle (or NULL)
   - PortIndexOut: optional out, zero-based port index, or PORT_INDEX_UNKNOWN if not present
   Returns:
     EFI_SUCCESS if we found a matching HC (PortIndexOut may be PORT_INDEX_UNKNOWN).
     EFI_NOT_FOUND if we couldn't find an HC prefix or device path.
*/
EFI_STATUS
FindHostControllerAndPortForUsbIo(
  IN EFI_HANDLE UsbIoHandle,
  OUT EFI_HANDLE *HcHandleOut,     /* optional */
  OUT UINT8 *PortIndexOut          /* optional: 0-based, or PORT_INDEX_UNKNOWN */
  )
{
  EFI_STATUS Status;
  EFI_DEVICE_PATH_PROTOCOL *DevPath = DevicePathFromHandle(UsbIoHandle);
  if (DevPath == NULL) {
    return EFI_NOT_FOUND;
  }

  if (HcHandleOut) *HcHandleOut = NULL;
  if (PortIndexOut) *PortIndexOut = PORT_INDEX_UNKNOWN;

  /* Try to find first USB device-path node and use ParentPortNumber if present */
  EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
  while (!IsDevicePathEnd(Node)) {
    if (DevicePathType(Node) == MESSAGING_DEVICE_PATH && DevicePathSubType(Node) == MSG_USB_DP) {
      USB_DEVICE_PATH_NODE *UsbNode = (USB_DEVICE_PATH_NODE *)Node;
      if (UsbNode->ParentPortNumber != 0) {
        if (PortIndexOut) *PortIndexOut = (UINT8)(UsbNode->ParentPortNumber - 1);
      }
      /* even if ParentPortNumber == 0, continue — we'll still try to find HC by prefix */
      break;
    }
    Node = NextDevicePathNode(Node);
  }

  /* Locate all USB host controllers and match device-path prefix */
  EFI_HANDLE *HcHandles = NULL;
  UINTN HcCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HcCount, &HcHandles);
  if (EFI_ERROR(Status) || HcCount == 0) {
    if (HcHandles) gBS->FreePool(HcHandles);
    return EFI_NOT_FOUND;
  }

  for (UINTN h = 0; h < HcCount; ++h) {
    EFI_DEVICE_PATH_PROTOCOL *HcPath = DevicePathFromHandle(HcHandles[h]);
    if (HcPath == NULL) continue;

    EFI_DEVICE_PATH_PROTOCOL *NextNode = NULL;
    if (IsDevicePathPrefix(HcPath, DevPath, &NextNode)) {
      /* We matched the HC path as a prefix of the child's path.
         This means HcHandles[h] is the host controller for the child. */
      if (HcHandleOut) *HcHandleOut = HcHandles[h];

      /* If the PortIndexOut wasn't filled from the USB node (== PORT_INDEX_UNKNOWN),
         we still return success since HC was located. Caller can decide how to proceed. */
      gBS->FreePool(HcHandles);
      return EFI_SUCCESS;
    }
  }

  if (HcHandles) gBS->FreePool(HcHandles);
  return EFI_NOT_FOUND;
}




#define PRINT_HEX16(x) (UINTN)(x)

STATIC
VOID
PrintGuidNameOrFull(
  IN EFI_GUID *Guid
  )
{
  if (Guid == NULL) {
    Print(L"    <null guid>\n");
    return;
  }
  /* %g prints a GUID in Print() */
  Print(L"    %g\n", Guid);
}

/* minimal USB config parser to print interfaces+endpoints */
STATIC
VOID
ParseAndPrintUsbConfig(
  IN VOID *CfgBuf,
  IN UINTN CfgLen
  )
{
  UINTN off = 0;
  while (off + 2 < CfgLen) {
    UINT8 bLength = ((UINT8 *)CfgBuf)[off + 0];
    UINT8 bDescriptorType = ((UINT8 *)CfgBuf)[off + 1];
    if (bLength < 2) break;
    if (off + bLength > CfgLen) break;

    if (bDescriptorType == 0x04) { /* Interface */
      UINT8 bInterfaceNumber = ((UINT8 *)CfgBuf)[off + 2];
      UINT8 bAlternateSetting = ((UINT8 *)CfgBuf)[off + 3];
      UINT8 bNumEndpoints = ((UINT8 *)CfgBuf)[off + 4];
      UINT8 bInterfaceClass = ((UINT8 *)CfgBuf)[off + 5];
      Print(L"      Interface %u alt %u endpoints=%u class=0x%02x\n",
            bInterfaceNumber, bAlternateSetting, bNumEndpoints, bInterfaceClass);
    } else if (bDescriptorType == 0x05) { /* Endpoint */
      UINT8 endpoint = ((UINT8 *)CfgBuf)[off + 2];
      UINT8 attributes = ((UINT8 *)CfgBuf)[off + 3];
      UINT16 maxpkt = (UINT16)(((UINT8 *)CfgBuf)[off + 4] | (((UINT8 *)CfgBuf)[off + 5] << 8));
      Print(L"        Endpoint 0x%02x attr=0x%02x maxpkt=%u\n", endpoint, attributes, maxpkt);
    }
    off += bLength;
  }
}

/* InspectHandle: list protocols, devicepath; if HC found, query capability & ports;
   if USB_IO found, dump descriptors. */
#if 1
EFI_STATUS
InspectHandle(
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;

  /* print handle pointer */
  Print(L"\n=== Inspect handle 0x%p ===\n", Handle);

  /* Device path text */
  EFI_DEVICE_PATH_PROTOCOL *DevPath = DevicePathFromHandle(Handle);
  if (DevPath) {
    CHAR16 *DpText = ConvertDevicePathToText(DevPath, TRUE, TRUE);
    if (DpText) {
      Print(L" DevicePath: %s\n", DpText);
      FreePool(DpText);
    }
  }

  /* list installed protocols */
  EFI_GUID **ProtocolBuffer = NULL;
  UINTN ProtocolCount = 0;
  Status = gBS->ProtocolsPerHandle(Handle, &ProtocolBuffer, &ProtocolCount);
  if (EFI_ERROR(Status)) {
    Print(L" ProtocolsPerHandle failed: 0x%08x\n", Status);
  } else {
    Print(L" Protocols installed: %u\n", ProtocolCount);
    for (UINTN i = 0; i < ProtocolCount; ++i) {
      Print(L"  "); PrintGuidNameOrFull(ProtocolBuffer[i]);
    }
  }

  /* If HC */
  {
    EFI_USB2_HC_PROTOCOL *Hc = NULL;
    Status = gBS->HandleProtocol(Handle, &gEfiUsb2HcProtocolGuid, (VOID **)&Hc);
    if (!EFI_ERROR(Status) && Hc != NULL) {
      UINT8 MaxSpeed = 0, NumPorts = 0, Is64 = 0;
      Status = Hc->GetCapability(Hc, &MaxSpeed, &NumPorts, &Is64);
      if (EFI_ERROR(Status)) {
        Print(L" GetCapability failed 0x%08x\n", Status);
      } else {
        Print(L" Host Controller: MaxSpeed=%u NumPorts=%u Is64=%u\n", MaxSpeed, NumPorts, Is64);
        for (UINT8 p = 1; p <= NumPorts; ++p) {
          EFI_USB_PORT_STATUS PortStatus;
          Status = Hc->GetRootHubPortStatus(Hc, p, &PortStatus);
          if (EFI_ERROR(Status)) {
            Print(L"  Port %u: GetRootHubPortStatus failed 0x%08x\n", p, Status);
            continue;
          }
          BOOLEAN present = (PortStatus.PortStatus & 0x01) != 0;
          Print(L"  Port %u: Present=%u Status=0x%08x Change=0x%08x\n",
                p, present ? 1 : 0, PortStatus.PortStatus, PortStatus.PortChangeStatus);
        }
      }
    }
  }

  /* If USB_IO */
  {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(Handle, &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
    if (!EFI_ERROR(Status) && UsbIo != NULL) {

      /* Device descriptor */
      EFI_USB_DEVICE_DESCRIPTOR DevDesc;
      Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
      if (EFI_ERROR(Status)) {
        Print(L" UsbGetDeviceDescriptor failed 0x%08x\n", Status);
      } else {
        Print(L" Device descriptor: VID=0x%04x PID=0x%04x Class=0x%02x NumConfigs=%u\n",
              DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass, DevDesc.NumConfigurations);
      }

      /* Try helper first */
      EFI_USB_CONFIG_DESCRIPTOR CfgHdr;
      ZeroMem(&CfgHdr, sizeof(CfgHdr));
      Status = UsbIo->UsbGetConfigDescriptor(UsbIo, &CfgHdr);
      if (!EFI_ERROR(Status)) {
        UINT16 totalLen = CfgHdr.TotalLength;
        if (totalLen < sizeof(EFI_USB_CONFIG_DESCRIPTOR)) totalLen = sizeof(EFI_USB_CONFIG_DESCRIPTOR);
        Print(L" UsbGetConfigDescriptor header reported TotalLength=%u\n", totalLen);
        VOID *CfgBuf = AllocateZeroPool(totalLen);
        if (CfgBuf) {
          EFI_USB_DEVICE_REQUEST Req;
          ZeroMem(&Req, sizeof(Req));
          Req.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
          Req.Request     = USB_REQ_GET_DESCRIPTOR;
          Req.Value       = (USB_DESC_TYPE_CONFIG << 8) | 0;
          Req.Index       = 0;
          Req.Length      = totalLen;
          UINT32 xferStatus = 0;
          Status = UsbIo->UsbControlTransfer(UsbIo, &Req, EfiUsbDataIn, (UINT32)DEFAULT_TIMEOUT_MS, CfgBuf, (UINTN)totalLen, &xferStatus);
          if (!EFI_ERROR(Status)) {
            EFI_USB_CONFIG_DESCRIPTOR *Full = (EFI_USB_CONFIG_DESCRIPTOR *)CfgBuf;
            Print(L" Full config: TotalLength=%u NumInterfaces=%u ConfigurationValue=%u\n",
                  Full->TotalLength, Full->NumInterfaces, Full->ConfigurationValue);
            ParseAndPrintUsbConfig(CfgBuf, totalLen);
          } else {
            Print(L" UsbControlTransfer(GET_DESCRIPTOR config) failed 0x%08x xfer=0x%08x\n", Status, xferStatus);
          }
          FreePool(CfgBuf);
        } else {
          Print(L" Failed to allocate %u bytes for config\n", totalLen);
        }
      } else {
        /* fallback GET_DESCRIPTOR(header)->GET_DESCRIPTOR(full) */
        Print(L" UsbGetConfigDescriptor failed 0x%08x, falling back\n", Status);
        EFI_USB_DEVICE_REQUEST ReqH;
        ZeroMem(&ReqH, sizeof(ReqH));
        ReqH.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
        ReqH.Request     = USB_REQ_GET_DESCRIPTOR;
        ReqH.Value       = (USB_DESC_TYPE_CONFIG << 8) | 0;
        ReqH.Index       = 0;
        ReqH.Length      = sizeof(EFI_USB_CONFIG_DESCRIPTOR);

        UINT8 HeaderBuf[sizeof(EFI_USB_CONFIG_DESCRIPTOR)];
        UINT32 hdrXfer = 0;
        Status = UsbIo->UsbControlTransfer(UsbIo, &ReqH, EfiUsbDataIn, (UINT32)DEFAULT_TIMEOUT_MS, HeaderBuf, (UINTN)sizeof(HeaderBuf), &hdrXfer);
        if (!EFI_ERROR(Status)) {
          UINT16 wTotal = (UINT16)HeaderBuf[2] | ((UINT16)HeaderBuf[3] << 8);
          if (wTotal < sizeof(EFI_USB_CONFIG_DESCRIPTOR)) wTotal = sizeof(EFI_USB_CONFIG_DESCRIPTOR);
          Print(L" Fallback header wTotalLength=%u\n", wTotal);
          VOID *FullBuf = AllocateZeroPool(wTotal);
          if (FullBuf) {
            EFI_USB_DEVICE_REQUEST ReqF;
            ZeroMem(&ReqF, sizeof(ReqF));
            ReqF.RequestType = USB_DEV_TO_HOST | USB_STANDARD | USB_DEVICE;
            ReqF.Request     = USB_REQ_GET_DESCRIPTOR;
            ReqF.Value       = (USB_DESC_TYPE_CONFIG << 8) | 0;
            ReqF.Index       = 0;
            ReqF.Length      = wTotal;
            UINT32 fullXfer = 0;
            Status = UsbIo->UsbControlTransfer(UsbIo, &ReqF, EfiUsbDataIn, (UINT32)DEFAULT_TIMEOUT_MS, FullBuf, (UINTN)wTotal, &fullXfer);
            if (!EFI_ERROR(Status)) {
              EFI_USB_CONFIG_DESCRIPTOR *Cfg = (EFI_USB_CONFIG_DESCRIPTOR *)FullBuf;
              Print(L" Fallback full config: TotalLength=%u NumInterfaces=%u ConfigurationValue=%u\n",
                    Cfg->TotalLength, Cfg->NumInterfaces, Cfg->ConfigurationValue);
              ParseAndPrintUsbConfig(FullBuf, wTotal);
            } else {
              Print(L" Fallback full GET_DESCRIPTOR failed 0x%08x xfer=0x%08x\n", Status, fullXfer);
            }
            FreePool(FullBuf);
          }
        } else {
          Print(L" Fallback header GET_DESCRIPTOR failed 0x%08x xfer=0x%08x\n", Status, hdrXfer);
        }
      }

      /* Optional: fetch string descriptors (manufacturer/product) if needed.
         Use UsbIo->UsbStringDescriptor or UsbControlTransfer with GET_DESCRIPTOR(TYPE_STRING).
         Keep in mind languages and buffer lengths. */
    }
  }

  if (ProtocolBuffer) {
    FreePool(ProtocolBuffer);
  }

  return EFI_SUCCESS;
}
#endif

///////////////// inspect handle ver 2 /////////////////////

//#define PCI_BAR0_OFFSET 0x10
//#define NUM_PCI_BARS 6
//#define PORT_INDEX_UNKNOWN 0xFF


/* Helper: find a handle that implements EFI_PCI_IO_PROTOCOL for the same device-path as Handle */
#if 0
EFI_STATUS
FindPciIoHandleForDevice(
  IN  EFI_HANDLE SourceHandle,
  OUT EFI_HANDLE *PciHandleOut  /* optional */
  )
{
  if (!PciHandleOut) return EFI_INVALID_PARAMETER;

  EFI_DEVICE_PATH_PROTOCOL *DevPath = DevicePathFromHandle(SourceHandle);
  if (DevPath == NULL) {
    return EFI_NOT_FOUND;
  }

  EFI_DEVICE_PATH_PROTOCOL *DpWalker = DevPath;
  EFI_HANDLE FoundHandle = NULL;
  EFI_STATUS Status = gBS->LocateDevicePath(&gEfiPciIoProtocolGuid, &DpWalker, &FoundHandle);
  if (!EFI_ERROR(Status) && FoundHandle != NULL) {
    *PciHandleOut = FoundHandle;
    return EFI_SUCCESS;
  }

  /* If not found, try scanning all handles that implement PCI_IO and match device path prefix */
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &Count, &Handles);
  if (EFI_ERROR(Status) || Count == 0) {
    if (Handles) gBS->FreePool(Handles);
    return EFI_NOT_FOUND;
  }

  for (UINTN i = 0; i < Count; ++i) {
    EFI_DEVICE_PATH_PROTOCOL *OtherDp = DevicePathFromHandle(Handles[i]);
    if (OtherDp == NULL) continue;
    EFI_DEVICE_PATH_PROTOCOL *next = NULL;
    /* simple prefix test: IsDevicePathPrefix is provided by DevicePathLib */
    if (IsDevicePathPrefix(DevPath, OtherDp, &next)) {
      *PciHandleOut = Handles[i];
      gBS->FreePool(Handles);
      return EFI_SUCCESS;
    }
    /* also check other way round: OtherDp prefix of DevPath */
    if (IsDevicePathPrefix(OtherDp, DevPath, &next)) {
      *PciHandleOut = Handles[i];
      gBS->FreePool(Handles);
      return EFI_SUCCESS;
    }
  }

  if (Handles) gBS->FreePool(Handles);
  return EFI_NOT_FOUND;
}

/* Print BARs (detect memory vs I/O; handle 64-bit BARs) */
EFI_STATUS
PrintPciBarsFromPciIo(
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
  if (PciIo == NULL) {
	  return EFI_INVALID_PARAMETER;
  }

  EFI_STATUS Status;
  UINT32 RawBarLow = 0;
  UINT64 BarVal64 = 0;
  for (UINTN bar = 0; bar < NUM_PCI_BARS; ++bar) {
    UINTN Offset = PCI_BAR0_OFFSET + (bar * 4);
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, Offset, 1, &RawBarLow);
    if (EFI_ERROR(Status)) {
      Print(L"  Read BAR%u failed (0x%08x)\n", (UINT32)bar, Status);
      continue;
    }

    /* If BAR == 0 or all 1s, treat as unused */
    if (RawBarLow == 0 || RawBarLow == 0xFFFFFFFF) {
      Print(L"  BAR%u: unused (raw=0x%08x)\n", (UINT32)bar, RawBarLow);
      continue;
    }

    /* I/O BAR if bit0 == 1 */
    if (RawBarLow & 1) {
      UINT32 IoAddr = RawBarLow & ~0x3u;
      Print(L"  BAR%u: I/O port (raw=0x%08x) -> I/O address = 0x%08x\n", (UINT32)bar, RawBarLow, IoAddr);
      continue;
    }

    /* Memory BAR */
    UINT32 typeBits = (RawBarLow >> 1) & 0x3; /* bits[2:1] */
    if (typeBits == 0x2) {
      /* 64-bit BAR (value = low DWORD here + next BAR gives high dword) */
      UINT32 RawBarHigh = 0;
      /* read next BAR high dword */
      Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, Offset + 4, 1, &RawBarHigh);
      if (EFI_ERROR(Status)) {
        Print(L"  BAR%u: 64-bit read high failed (0x%08x)\n", (UINT32)bar, Status);
        continue;
      }
      BarVal64 = ((UINT64)RawBarHigh << 32) | (UINT64)(RawBarLow & ~0xFULL); /* mask low flags */
      Print(L"  BAR%u: MMIO 64-bit (raw low=0x%08x high=0x%08x) -> addr=0x%016llx\n",
            (UINT32)bar, RawBarLow, RawBarHigh, BarVal64);
      bar++; /* skip next BAR since consumed */
    } else {
      /* 32-bit MMIO */
      UINT64 Addr32 = (UINT64)(RawBarLow & ~0xFULL);
      Print(L"  BAR%u: MMIO 32-bit (raw=0x%08x) -> addr=0x%08x\n", (UINT32)bar, RawBarLow, (UINT32)Addr32);
    }
  }
  return EFI_SUCCESS;
}

/* Find PCI IO for a handle (using device path) and print BARs */
EFI_STATUS
ShowPciDeviceBarsForHandle(
  IN EFI_HANDLE SourceHandle
  )
{
  EFI_STATUS Status;
  EFI_HANDLE PciHandle = NULL;
  Status = FindPciIoHandleForDevice(SourceHandle, &PciHandle);
  if (EFI_ERROR(Status) || PciHandle == NULL) {
    Print(L"Could not find PCI_IO handle for %p (Status=0x%08x)\n", SourceHandle, Status);
    return Status;
  }

  Print(L"Located PCI_IO handle %p for source %p\n", PciHandle, SourceHandle);

  EFI_PCI_IO_PROTOCOL *PciIo = NULL;
  Status = gBS->HandleProtocol(PciHandle, &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
  if (EFI_ERROR(Status) || PciIo == NULL) {
    Print(L" HandleProtocol(PCI_IO) failed (0x%08x)\n", Status);
    return Status;
  }

  /* print PCI location if GetLocation exists */
  UINTN Segment=0, Bus=0, Device=0, Function=0;
  if (PciIo->GetLocation) {
    Status = PciIo->GetLocation(PciIo, &Segment, &Bus, &Device, &Function);
    if (!EFI_ERROR(Status)) {
      Print(L" PCI location: Seg=%u Bus=%u Dev=%u Func=%u\n", Segment, Bus, Device, Function);
    }
  }

  /* Print BARs/MMIO addresses */
  PrintPciBarsFromPciIo(PciIo);
  return EFI_SUCCESS;
}
#endif





/* Main: run high-level then manual safe on root port index 0 */
EFI_STATUS
EFIAPI
UefiMain(
  IN EFI_HANDLE ImageHandle,
  IN EFI_SYSTEM_TABLE *SystemTable
  )
{
  EFI_STATUS Status;
  
  // setup GOP
	Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID **)&mGraphicsOuput);
	SystemTable->ConOut->SetCursorPosition(SystemTable->ConOut, 0, 0);
	UINTN gop_querymode_size = sizeof(EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
	EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *mode_info = NULL;
	Status = mGraphicsOuput->QueryMode(mGraphicsOuput, mGraphicsOuput->Mode->Mode, 
		&gop_querymode_size, &mode_info); 

	mGraphicsOuput->Blt(mGraphicsOuput, &white, EfiBltVideoFill, 0, 0, 0, 0, 
			mGraphicsOuput->Mode->Info->HorizontalResolution, mGraphicsOuput->Mode->Info->VerticalResolution, 0);
  

  Print(L"\nUSB enumeration (safe manual mode)\n");

  /* show existing devices */
  EnumerateViaUsbIo();
  Print(L"\n");

// VER 1
  /* find host controllers */
  /*
  EFI_HANDLE *HcHandles = NULL;
  UINTN HcCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HcCount, &HcHandles);
  if (EFI_ERROR(Status) || HcCount == 0) {
    Print(L"No EFI_USB2_HC_PROTOCOL handles found (0x%08x)\n", Status);
    return EFI_NOT_FOUND;
  }
  Print(L"Found %u host controllers\n", HcCount);

  for (UINTN i = 0; i < HcCount; ++i) {
    EFI_USB2_HC_PROTOCOL *Hc = NULL;
    Status = gBS->HandleProtocol(HcHandles[i], &gEfiUsb2HcProtocolGuid, (VOID **)&Hc);
    if (EFI_ERROR(Status) || Hc == NULL) continue;

    UINT8 MaxSpeed=0, NumPorts=0, Is64=0;
    Status = Hc->GetCapability(Hc, &MaxSpeed, &NumPorts, &Is64);
    if (!EFI_ERROR(Status)) {
      Print(L" Host[%u] MaxSpeed=%u NumPorts=%u\n", i, NumPorts, MaxSpeed);
    }

    if (NumPorts >= 1) {
      Print(L" Inspecting root port 1 (index 0) of host %u\n", i);
      ManualEnumeratePortSafe(Hc, 0);
    }
  }

  if (HcHandles) gBS->FreePool(HcHandles);
  return EFI_SUCCESS;
*/  
  
  
  // VER 2
  
    /* 2) If there are devices, pick the first one and get its handle & parent port */
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &Count, &Handles);
  if (!EFI_ERROR(Status) && Count > 0) {
    EFI_HANDLE firstHandle = Handles[0];
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(firstHandle, &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
    if (!EFI_ERROR(Status) && UsbIo != NULL) {
      EFI_USB_DEVICE_DESCRIPTOR DevDesc;
      Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &DevDesc);
      if (!EFI_ERROR(Status)) {
        Print(L"Picked Handle[0] VID=0x%04x PID=0x%04x\n", DevDesc.IdVendor, DevDesc.IdProduct);

        /* Get parent port number (1-based) */
        /*
		UINT8 ParentPort = 0;
        Status = GetParentPortFromUsbIoHandle(firstHandle, &ParentPort);
		 if (!EFI_ERROR(Status)) {
          Print(L" ParentPortNumber (UEFI device-path, 1-based) = %u\n", ParentPort);
          Print(L" Convert to zero-based portIndex = %u\n", (UINT32)(ParentPort - 1));
        } else {
          Print(L" Could not extract parent-port from device path (0x%08x)\n", Status);
        }
		*/
		
// first examine test - list of protocols 
#if 0
		EFI_HANDLE MatchedHc = NULL;
		UINT8 PortIndex = 0;
		Status = FindHostControllerAndPortForUsbIo(firstHandle, &MatchedHc, &PortIndex);
		if (!EFI_ERROR(Status)) {
		  Print(L" Found HC handle 0x%p, portIndex (0-based) = %u\n", MatchedHc, (UINT32)PortIndex);
		} else {
		  Print(L" Could not determine parent port (Status=0x%08x)\n", Status);
		}
#endif

// second examine test - list of protocols 
#if 0
		EFI_HANDLE MatchedHc = NULL;
		UINT8 PortIndex = PORT_INDEX_UNKNOWN; /* 0xFF */
		Status = FindHostControllerAndPortForUsbIo(firstHandle, &MatchedHc, &PortIndex);
		if (!EFI_ERROR(Status)) {
		  if (MatchedHc) {
			Print(L"Found matching HC handle 0x%p\n", MatchedHc);
		  } else {
			Print(L"HC not returned (unexpected)\n");
		  }
		  if (PortIndex != PORT_INDEX_UNKNOWN) {
			Print(L" Parent port (0-based) = %u\n", (UINT32)PortIndex);
		  } else {
			Print(L" Parent port unknown (device-path USB node had 0)\n");
		  }
		} else {
		  Print(L"Could not match UsbIo handle to a host controller (Status=0x%08x)\n", Status);
		}
#endif

	InspectHandle((EFI_HANDLE)(UINTN)0x7DFD4018); /* host controller handle 161 */
	InspectHandle((EFI_HANDLE)(UINTN)0x7DF59A98); /* USB device handle 170 (or 209 in other run) */
		
	// seconde test for handles
	//InspectTwoHandlesPrintBars();
       

        /* Dump descriptors using UsbIo */
        DumpUsbIoDescriptors(firstHandle);
      }
    }

    gBS->FreePool(Handles);
  } else {
    Print(L"No UsbIo handles found initially; attempting to find one by VID/PID as demo.\n");

    /* Example: poll for a known VID/PID (replace with your VID/PID if you know it) */
    UINT16 wishVid = 0x058F; /* example from your logs */
    UINT16 wishPid = 0x6387;

    EFI_HANDLE found = NULL;
    Status = FindUsbIoByVidPid(wishVid, wishPid, IO_POLL_RETRIES, IO_POLL_INTERVAL_MS, &found);
    if (!EFI_ERROR(Status)) {
      Print(L"Found UsbIo for VID=0x%04x PID=0x%04x via polling\n", wishVid, wishPid);

      UINT8 ParentPort = 0;
      if (!EFI_ERROR(GetParentPortFromUsbIoHandle(found, &ParentPort))) {
        Print(L" ParentPortNumber = %u (1-based)\n", ParentPort);
      } else {
        Print(L" Could not read parent port from device path\n");
      }

      DumpUsbIoDescriptors(found);
    } else {
      Print(L"Timed out waiting for device VID=0x%04x PID=0x%04x (Status=0x%08x)\n", wishVid, wishPid, Status);
    }
  }
  
  
  // wait for key
  Print(L"Done. Press any key to exit...\n");
  EFI_INPUT_KEY Key;
  while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) != EFI_SUCCESS) {
    // spin
  }

  return EFI_SUCCESS;
  
}
