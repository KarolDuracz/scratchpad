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

/* Utility: search existing UsbIo handles for a device by VID/PID.
   Polls a few times until timeout to allow bus driver to create handle. */
STATIC
EFI_STATUS
FindUsbIoByVidPid(
  IN UINT16 Vid,
  IN UINT16 Pid,
  IN UINTN  PollRetries,
  IN UINTN  PollIntervalMs,
  OUT EFI_HANDLE *FoundHandle  /* optional out */
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

/* High-level enumeration (same as before) */
STATIC
EFI_STATUS
EnumerateViaUsbIo(VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;

  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &Count, &Handles);
  if (EFI_ERROR(Status) || Count == 0) {
    Print(L"No existing EFI_USB_IO handles found (Status=0x%08x)\n", Status);
    if (Handles) gBS->FreePool(Handles);
    return EFI_NOT_FOUND;
  }

  Print(L"Found %u USB device handle(s)\n", Count);
  for (UINTN i = 0; i < Count; ++i) {
    EFI_USB_IO_PROTOCOL *UsbIo = NULL;
    Status = gBS->HandleProtocol(Handles[i], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
    if (EFI_ERROR(Status) || UsbIo == NULL) continue;
    EFI_USB_DEVICE_DESCRIPTOR Desc;
    Status = UsbIo->UsbGetDeviceDescriptor(UsbIo, &Desc);
    if (EFI_ERROR(Status)) continue;
    Print(L" Handle[%u] VID=0x%04x PID=0x%04x\n", i, Desc.IdVendor, Desc.IdProduct);
  }
  gBS->FreePool(Handles);
  return EFI_SUCCESS;
}

/* Main: run high-level then manual safe on root port index 0 */
EFI_STATUS
EFIAPI
UefiMain(
  IN EFI_HANDLE ImageHandle,
  IN EFI_SYSTEM_TABLE *SystemTable
  )
{
  EFI_STATUS Status;

  Print(L"\nUSB enumeration (safe manual mode)\n");

  /* show existing devices */
  EnumerateViaUsbIo();
  Print(L"\n");

  /* find host controllers */
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
}
