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
#include <Protocol/BlockIo.h>
#include <Guid/GlobalVariable.h>
#include <IndustryStandard/Usb.h>

#include <Protocol/DevicePath.h>

#include <Protocol/DriverBinding.h>
#include <Protocol/LoadedImage.h>

// Add at top if not present
#include <Library/DevicePathLib.h> // optional if you rely on devicepath helpers


#include <IndustryStandard/Usb.h>   // for USB endpoint type macros (if available)



#define DP_TYPE_END            0x7F
#define DP_SUBTYPE_END_ENTIRE  0xFF

#define DP_TYPE_HARDWARE       0x01
#define DP_TYPE_MESSAGING      0x03

#define DP_SUBTYPE_PCI         0x01   // PCI device path subtype (Hardware)
#define DP_SUBTYPE_USB         0x05   // USB device path subtype (Messaging)

#define MAX_PORTS_SAFE 32   // sanity cap for ports to avoid runaway loops

#define TIMEOUT_MS 5000
#define MAX_BARS 6
#define MAX_PORTS_TO_PRINT 16

//#define MAX_BARS 6
//#define MAX_PORTS_SAFE 32
#define FRINDEX_POLL_COUNT 5
#define FRINDEX_POLL_DELAY_MS 50

//
#define USB_ENDPOINT_TYPE_CONTROL     0x00
#define USB_ENDPOINT_TYPE_ISOCHRONOUS 0x01
#define USB_ENDPOINT_TYPE_BULK        0x02
#define USB_ENDPOINT_TYPE_INTERRUPT   0x03



CHAR16 *
EFIAPI
MyDevicePathToStr (
  IN CONST EFI_DEVICE_PATH_PROTOCOL  *DevicePath
  )
{
  CONST EFI_DEVICE_PATH_PROTOCOL *Node;
  CHAR16 *Out;
  UINTN OutSizeChars = 1024;
  UINTN Pos = 0;

  if (DevicePath == NULL) {
    return NULL;
  }

  Out = AllocatePool (OutSizeChars * sizeof(CHAR16));
  if (Out == NULL) {
    return NULL;
  }
  Out[0] = L'\0';

  Node = DevicePath;
  while (Node != NULL) {
    UINT8 Type = Node->Type;
    UINT8 SubType = Node->SubType;
    UINT16 NodeLen = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));

    // End node?
    if (Type == DP_TYPE_END && SubType == DP_SUBTYPE_END_ENTIRE) {
      break;
    }
    // Guard: minimal node length is 4 (header)
    if (NodeLen < 4) {
      break;
    }

    // Ensure space (characters) for next append; if low, grow buffer
    if ((OutSizeChars - Pos) < 64) {
      UINTN NewSize = OutSizeChars * 2;
      CHAR16 *New = AllocatePool (NewSize * sizeof(CHAR16));
      if (New == NULL) break;
      CopyMem (New, Out, OutSizeChars * sizeof(CHAR16));
      FreePool (Out);
      Out = New;
      OutSizeChars = NewSize;
    }

    // Remaining size in bytes for UnicodeSPrint (it expects bytes)
    UINTN RemainingBytes = (OutSizeChars - Pos) * sizeof(CHAR16);

    if (Type == DP_TYPE_HARDWARE && SubType == DP_SUBTYPE_PCI && NodeLen >= 6) {
      // PCI node layout: Header (4) + UINT8 Device + UINT8 Function
      UINT8 *raw = (UINT8*)Node;
      UINT8 Device = raw[4];
      UINT8 Function = raw[5];
      // append "Pci(d,f)/"
      UnicodeSPrint (&Out[Pos], RemainingBytes, L"Pci(%u,%u)/", (UINTN)Device, (UINTN)Function);
      Pos += StrLen (&Out[Pos]);
    } else if (Type == DP_TYPE_MESSAGING && SubType == DP_SUBTYPE_USB && NodeLen >= 6) {
      // USB node layout: Header (4) + UINT8 ParentPortNumber + UINT8 InterfaceNumber
      UINT8 *raw = (UINT8*)Node;
      UINT8 ParentPort = raw[4];
      UINT8 Interface = raw[5];
      // append "USB(port,iface)/"
      UnicodeSPrint (&Out[Pos], RemainingBytes, L"USB(%u,%u)/", (UINTN)ParentPort, (UINTN)Interface);
      Pos += StrLen (&Out[Pos]);
    } else {
      // Generic node print: "Txx.Syy/"
      UnicodeSPrint (&Out[Pos], RemainingBytes, L"T%02x.S%02x/", (UINTN)Type, (UINTN)SubType);
      Pos += StrLen (&Out[Pos]);
    }

    // advance to next node
    Node = (CONST EFI_DEVICE_PATH_PROTOCOL *) ((CONST UINT8 *) Node + NodeLen);
  }

  // Ensure not empty
  if (Pos == 0) {
    UnicodeSPrint (Out, OutSizeChars * sizeof(CHAR16), L"(empty device path)");
  }

  return Out;
}

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




/*
		This is part to examine why those all registers are zeros
		
		 --- Diagnose: controller BAR0 OpBase=0x010 port=1 ---
    USBCMD=0x00000000 USBSTS=0x00000000 FRINDEX=0x00000000
    ASYNCLISTADDR=0x00000000 PERIODICLISTBASE=0      PORT1: PORTSC raw=0x00010101  (CCS=1 CSC=0 PE=0 PEchg=0 PO=0 Speed=0)
    NOTE: Host controller Run/Stop == 0 (controller stopped). Driver may not have started schedules.
    NOTE: ASYNCLISTADDR == 0 (no async list programmed).
    NOTE: Async schedule not enabled (USBCMD.AsyncEnable=0).
    NOTE: Periodic schedule not enabled (USBCMD.PeriodicEnable=0).
    Could not determine HC protocol handle by location (best-effort failed).
    Suggestion: ensure HC DXE driver    Result: connected, not enabled, EHCI owns port -> likely HC scheduling/driver not started.
    Next safe diagnostic steps: (1) verify HC driver is present in DXE image; (2) enable driver debug logging; (3) check for companion controller and driver.
    If you control firmware build: ensure UsbBusDxe and relevant host-controller drivers are included and started.
  --- End diagnose for port 1 ---
Controller PCI BAR0: Found port 2 wi  No HC protocol handle found for this PCI device (driver may not have bound)
  --- Diagnose: controller BAR0 OpBase=0x010 port=2 ---
    USBCMD=0x00000000 USBSTS=0x00000000 FRINDEX=0x00000000
    ASYNCLISTADDR=0x00000000 PERIODICLISTBASE=0x00000000 HCSPARAMS=0x00000200
      PORT2: PORTSC raw=0x00010101      NOTE: Host controller Run/Stop == 0 (controller stopped). Driver may not have started schedules.
    NOTE: ASYNCLISTADDR == 0 (no async list programmed).
    NOTE: Async schedule not enabled (USBCMD.AsyncEnable=0).
    NOTE: Periodic schedule not enabled (USBCMD.PeriodicEnable=0).
    Could not determine HC protocol handle by location (best-effort failed).
    Suggestion: ensure HC DXE driver (EHCI/OHCI/UHCI) for this chipset is present in your firmware build.
    Result: connected, not enabled, EHCI owns port -> likely HC scheduling/driver not started.
    Next safe diagnostic steps: (1) verify HC driver is present in DXE image; (2) enable driver debug logging; (3) check for companion controller and driver.
    If you control firmware build: ensure UsbBusDxe and relevant host-controller drivers are included and started.
  --- End diagnose for port 2 ---


*/

//
// Diagnostic helper: print PCI Command, BARs, BAR types and attempt safe reads.
// No writes. Defensive against errors/NULLs.
//
// Requires: <Uefi.h>, <Library/UefiLib.h>, <Library/UefiBootServicesTableLib.h>,
//           <Protocol/PciIo.h>, <Library/MemoryAllocationLib.h>, <Library/BaseMemoryLib.h>
//

#define DIAG_BAR_READ_BYTES 64   // how many bytes to attempt to read from MMIO BAR
//#define MAX_BARS 6

#define PCI_IO_READ_MEM8(pi, bar, off, cnt, buf) \
  (pi)->Mem.Read((pi), EfiPciIoWidthUint8, (UINT8)(bar), (UINT64)(off), (UINTN)(cnt), (buf))

#define PCI_IO_READ_IO8(pi, bar, off, cnt, buf) \
  (pi)->Io.Read((pi), EfiPciIoWidthUint8, (UINT8)(bar), (UINT64)(off), (UINTN)(cnt), (buf))

// optional call
// EFI_STATUS r = PCI_IO_READ_IO8(PciIo, BarIndex, IoBase, sizeof(Buf8), Buf8);

STATIC
VOID
PrintPciCommandBits (
  IN UINT16 Cmd
  )
{
  Print (L"    PCI Command = 0x%04x : MEM=%u IO=%u BUSMASTER=%u INTXDIS=%u\n",
         Cmd,
         (UINT32)((Cmd >> 1) & 1),
         (UINT32)(Cmd & 1),
         (UINT32)((Cmd >> 2) & 1),
         (UINT32)((Cmd >> 10) & 1));
}

STATIC
VOID
DiagPciBarAndMmio (
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
	
	/////////////////////////////////
	////////////////////////////////
	// for real HW I don't need this.
	// just exit every time when it call to it
	
  if (PciIo == NULL) {
    Print (L"DiagPciBarAndMmio: PciIo == NULL\n");
    return;
  }

  EFI_STATUS Status;
  UINTN Segment, Bus, Device, Function;
  PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);
  Print (L"=== Diag for PCI %02x:%02x.%x ===\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function);

  // Read PCI Command/Status (offset 0x04) and HeaderType
  UINT32 Dword = 0;
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x04, 1, &Dword);
  if (EFI_ERROR (Status)) {
    Print (L"  Failed to read PCI dword @0x04: 0x%lx\n", Status);
    return;
  }
  UINT16 Command = (UINT16)(Dword & 0xFFFF);
  UINT16 StatusReg = (UINT16)((Dword >> 16) & 0xFFFF);
  PrintPciCommandBits (Command);
  Print (L"    PCI Status = 0x%04x\n", StatusReg);

  // Read HeaderType
  UINT8 HeaderType = 0;
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint8, 0x0E, 1, &HeaderType);
  if (!EFI_ERROR (Status)) {
    Print (L"    PCI HeaderType = 0x%02x\n", HeaderType);
  }

  // Inspect BARs at offsets 0x10,0x14,...0x24 (up to 6 BARs)
  for (UINTN BarIndex = 0; BarIndex < MAX_BARS; ++BarIndex) {
    UINTN Offset = 0x10 + (UINTN)BarIndex * 4;
    UINT32 BarLo = 0;
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, (UINT32)Offset, 1, &BarLo);
    if (EFI_ERROR (Status)) {
      Print (L"  BAR%u: PCI read failed at offset 0x%x: 0x%lx\n", (UINT32)BarIndex, (UINT32)Offset, Status);
      continue;
    }

    if (BarLo == 0) {
      Print (L"  BAR%u: raw=0x%08x (empty)\n", (UINT32)BarIndex, BarLo);
      // still try next BAR (some devices use fewer BARs)
      continue;
    }

    // Check IO vs Memory: bit0 set => I/O space
    if (BarLo & 1) {
      // I/O space BAR
      UINT32 IoBase = BarLo & ~0x3;
      Print (L"  BAR%u: I/O space raw=0x%08x  IoBase=0x%08x\n", (UINT32)BarIndex, BarLo, IoBase);
      // Try to read first 8 bytes with PciIo->Io.Read (safe)
      //UINT8 Buf8[16];
      //EFI_STATUS r = PciIo->Io.Read (PciIo, EfiPciIoWidthUint8, BarIndex, IoBase, sizeof(Buf8), Buf8);
	  
	  UINT8 Buf8[16];
		EFI_STATUS r = PciIo->Io.Read (
                          PciIo,
                          EfiPciIoWidthUint8,
                          (UINT8)BarIndex,           // BarIndex is small (cast for prototype)
                          (UINT64)IoBase,            // Offset must be 64-bit
                          (UINTN)sizeof(Buf8),       // Count is size_t / UINTN
                          Buf8
                          );

	  
      if (!EFI_ERROR (r)) {
        Print (L"    Read via Io.Read succeeded (first 16 bytes):\n");
        DumpHex (L"    IO BAR data", Buf8, sizeof(Buf8));
      } else {
        Print (L"    Io.Read failed: 0x%lx\n", r);
      }
      continue;
    } else {
      // Memory BAR: determine type
      UINT32 type = (BarLo >> 1) & 0x3; // 00=32-bit, 10=64-bit
      BOOLEAN Is64 = (type == 2);
      UINT64 BarAddr = (UINT64)(BarLo & ~0xF); // mask low bits (type/prefetch)
      if (Is64) {
        // read high dword
        UINT32 BarHigh = 0;
        //Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, Offset + 4, 1, &BarHigh);
        
		// Offset is a UINTN; Pci.Read expects a 32-bit config offset parameter in many headers.
		// Cast explicitly toUINT32 to silence narrowing warning.
		Status = PciIo->Pci.Read (
                          PciIo,
                          EfiPciIoWidthUint32,
                          (UINT32)(Offset + 4),
                          1,
                          &BarHigh
                          );

		
		if (!EFI_ERROR (Status)) {
          BarAddr |= ((UINT64)BarHigh) << 32;
        }
      }
      Print (L"  BAR%u: MMIO raw=0x%08x type=%s  base=0x%016lx\n",
             (UINT32)BarIndex, BarLo, Is64 ? L"64-bit" : L"32-bit", (UINT64)BarAddr);

      // Now check whether PCI Command Memory Space Enable is set
      if (((Command >> 1) & 1) == 0) {
        Print (L"    WARNING: PCI Command.MemorySpaceEnable == 0 ; device's MMIO BAR is not decoded by bus.\n");
        Print (L"             Many MMIO reads will return 0 until the controller is enabled.\n");
        // still attempt a read; it will likely be zero.
      }

      // Attempt to read the first DIAG bytes from this BAR via PciIo->Mem.Read
      // Note: PciIo uses BAR index (not physical address) for Mem.Read.
      UINT8 *Buf = AllocatePool (DIAG_BAR_READ_BYTES);
      if (Buf == NULL) {
        Print (L"    Failed to allocate buffer for MMIO read\n");
      } else {
        SetMem (Buf, DIAG_BAR_READ_BYTES, 0xEE);
        //EFI_STATUS r = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x0, DIAG_BAR_READ_BYTES, Buf);
		
			EFI_STATUS r = PciIo->Mem.Read (
                          PciIo,
                          EfiPciIoWidthUint8,
                          (UINT8)BarIndex,         // cast to expected small type
                          (UINT64)0x0,             // offset = 0 (64-bit)
                          (UINTN)DIAG_BAR_READ_BYTES,
                          Buf
                          );


		
        if (EFI_ERROR (r)) {
          Print (L"    Mem.Read (BarIndex=%u) failed: 0x%lx\n", (UINT32)BarIndex, r);
        } else {
          // if the entire buffer is zero or 0xEE, note that
          BOOLEAN allZero = TRUE;
          BOOLEAN allEE   = TRUE;
          for (UINTN k = 0; k < DIAG_BAR_READ_BYTES; ++k) {
            if (Buf[k] != 0) { allZero = FALSE; }
            if (Buf[k] != 0xEE) { allEE = FALSE; }
          }
          if (allZero) {
            Print (L"    Mem.Read returned all zeros for first %u bytes (suspicious).\n", DIAG_BAR_READ_BYTES);
          } else if (allEE) {
            Print (L"    Mem.Read returned unchanged pattern (0xEE) -> read likely failed to fill buffer.\n");
          } else {
            Print (L"    Mem.Read succeeded, dump first %u bytes:\n", DIAG_BAR_READ_BYTES);
            DumpHex (L"    MMIO BAR data", Buf, DIAG_BAR_READ_BYTES);
          }
        }
        FreePool (Buf);
      }
    } // memory BAR branch
  } // for each BAR

  // Suggest next actions based on findings
  Print (L"  Suggestions:\n");
  Print (L"    * If MemorySpaceEnable==0: enable MMIO in platform (or ensure HC driver sets it). Do NOT write PCI config in production without care.\n");
  Print (L"    * If BAR is I/O space, try Io.Read as above. EHCI normally uses MMIO; I/O BAR may indicate legacy controller mode.\n");
  Print (L"    * If MMIO reads are all zeros but BAR base != 0 and MemorySpaceEnable==1: device may be in reset or power-gated - check platform power/ACPI settings or HC driver.\n");
  Print (L"    * Check that host controller DXE driver is included (EhciDxe/UsbBusDxe etc.). A missing driver often results in no schedule programmed.\n");

  Print (L"=== End Diag for PCI %02x:%02x.%x ===\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function);
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

	// extra test
	//DiagPciBarAndMmio (PciIo);

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
  Map EFI_USB2_HC_PROTOCOL handles to PCI B:D.F (if PciIo available on same handle)
**/
STATIC
VOID
MapUsb2HcToPciBdf (VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *HcHandles = NULL;
  UINTN HcCount = 0;
  UINTN i;

  Print (L"--- Mapping USB2_HC handles to PCI B:D.F ---\n");
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HcCount, &HcHandles);
  if (EFI_ERROR (Status) || HcCount == 0) {
    Print (L"  No USB2_HC handles: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HcCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    UINTN Bus, Device, Function, Segment;

    Status = gBS->HandleProtocol (HcHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (!EFI_ERROR (Status) && PciIo != NULL) {
      PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);
      Print (L"  USB2_HC handle %u -> PCI %02x:%02x.%x (Segment=%u)\n", (UINT32)i, (UINT32)Bus, (UINT32)Device, (UINT32)Function, (UINT32)Segment);
    } else {
      Print (L"  USB2_HC handle %u -> PciIo not on same handle (or not accessible): 0x%lx\n", (UINT32)i, Status);
    }
	
	// extra test
	//DiagPciBarAndMmio (PciIo);
	
  }

  if (HcHandles) {
    FreePool (HcHandles);
  }
}

/**
  Probe PCI handle BARs for EHCI capability + operational registers and print key registers.
  Read-only; does not write registers.
**/
STATIC
VOID
CheckEhciRegistersForPciHandle (
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;
  EFI_PCI_IO_PROTOCOL *PciIo;
  UINT32 Data32;
  UINTN Bus, Device, Function, Segment;
  //UINTN BarIndex;
  UINT8 BarIndex;

  Status = gBS->HandleProtocol (Handle, &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
  if (EFI_ERROR (Status)) {
    return;
  }

  PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);

  // Read DWORD at 0x08 (Class/Subclass/ProgIf/RevID)
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &Data32);
  if (EFI_ERROR (Status)) {
    return;
  }
  
  // extra test
  //DiagPciBarAndMmio (PciIo);
  
  UINT8 ClassCode = (UINT8)((Data32 >> 24) & 0xFF);
  UINT8 SubClass  = (UINT8)((Data32 >> 16) & 0xFF);
  UINT8 ProgIf    = (UINT8)((Data32 >> 8) & 0xFF);

  if (!(ClassCode == 0x0C && SubClass == 0x03)) {
    // Not a USB controller
    return;
  }

  Print (L"--- EHCI probe for PCI %02x:%02x.%x | PROGIF %02x ---\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function, (UINT32)ProgIf);

  for (BarIndex = 0; BarIndex < MAX_BARS; ++BarIndex) {
    UINT8 CapLength = 0xFF;
    // Try reading the first byte of BAR region (cap length)
    Status = PciIo->Mem.Read (
                       PciIo,
                       EfiPciIoWidthUint8,
                       BarIndex,
                       0x00,
                       1,
                       &CapLength
                       );
    if (EFI_ERROR (Status)) {
      continue;
    }

    if (CapLength == 0xFF || CapLength == 0x00) {
      continue;
    }

    UINT16 HciVersion = 0;
    Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint16, BarIndex, 0x02, 1, &HciVersion);
    if (EFI_ERROR (Status)) {
      continue;
    }

    Print (L"  BAR%u: CAPLENGTH=0x%02x HCIVERSION=0x%04x\n", (UINT32)BarIndex, CapLength, HciVersion);

    UINTN OpBase = (UINTN)CapLength;

    #define READ_OP32(off, var) \
      do { \
        EFI_STATUS _st = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, (off) + OpBase, 1, &(var)); \
        if (EFI_ERROR (_st)) { Print (L"    Read op @0x%03x failed: 0x%lx\n", (UINT32)((off)+OpBase), _st); } \
      } while (0)

    UINT32 UsbCmd = 0, UsbSts = 0, UsbIntr = 0, FrIndex = 0, AsyncList = 0, PeriodicBase = 0;
    READ_OP32 (0x00, UsbCmd);       // USBCMD offset 0x00
    READ_OP32 (0x04, UsbSts);       // USBSTS offset 0x04
    READ_OP32 (0x08, UsbIntr);      // USBINTR offset 0x08
    READ_OP32 (0x0C, FrIndex);      // FRINDEX offset 0x0C
    READ_OP32 (0x18, AsyncList);    // ASYNCLISTADDR offset 0x18
    READ_OP32 (0x10, PeriodicBase); // PERIODICLISTBASE offset 0x10

    Print (L"    USBCMD=0x%08x USBSTS=0x%08x USBINTR=0x%08x FRINDEX=0x%08x\n", UsbCmd, UsbSts, UsbIntr, FrIndex);
    Print (L"    PERIODICLISTBASE=0x%08x ASYNCLISTADDR=0x%08x\n", PeriodicBase, AsyncList);

    // Read HCSPARAMS (capability offset 0x04)
    UINT32 HcSParams = 0;
    Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);
    if (!EFI_ERROR (Status)) {
      // display raw and approximate port count (lowest 4 bits often used)
      UINT32 NPorts = (HcSParams & 0x0000000F);
      Print (L"    HCSPARAMS(raw)=0x%08x  (N_PORTS approx=%u)\n", HcSParams, NPorts);
    } else {
      Print (L"    HCSPARAMS read failed: 0x%lx\n", Status);
    }

    // Read PORTSC registers: operational offset 0x44 + 4 * port
    UINTN PortIndex;
    for (PortIndex = 0; PortIndex < MAX_PORTS_TO_PRINT; ++PortIndex) {
      UINT32 PortSc = 0;
      UINTN PortOffset = OpBase + 0x44 + (UINTN)PortIndex * 4;
      EFI_STATUS _r = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
      if (EFI_ERROR (_r)) {
        // stop if reading next port fails
        break;
      }
      Print (L"      PORT%u: PORTSC raw=0x%08x  (CCS=%d CSC=%d PE=%d PortSpeed=%d PO=%d)\n",
             (UINT32)PortIndex,
             PortSc,
             (int)((PortSc >> 0) & 1),         // Current Connect Status
             (int)((PortSc >> 1) & 1),         // Connect Status Change
             (int)((PortSc >> 2) & 1),         // Port Enabled
             (int)((PortSc >> 26) & 3),        // simplified port speed decode (spec varies)
             (int)((PortSc >> 13) & 1)         // Port Owner (example bit position)
             );
    }

    // stop after first viable EHCI region found
    break;
  } // for each BAR
}

/**
  Scan all PCI handles and run the EHCI probe for USB controllers.
**/
STATIC
VOID
ScanAllPciForEhciAndCheck (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN i;

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(PciIo) failed: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HandleCount; ++i) {
    CheckEhciRegistersForPciHandle (Handles[i]);
  }

  if (Handles) {
    FreePool (Handles);
  }
}

/**
  Find BlockIo handles and read LBA0 (safe read) for devices where MediaPresent is true.
**/
STATIC
VOID
CheckUsbBlockIoAndReadLba0 (VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN i;

  Print (L"--- Stage: Check BlockIo on USB mass-storage devices ---\n");

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiBlockIoProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR (Status) || HandleCount == 0) {
    Print (L"  No BlockIo handles found: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HandleCount; ++i) {
    EFI_BLOCK_IO_PROTOCOL *Blk = NULL;
    Status = gBS->HandleProtocol (Handles[i], &gEfiBlockIoProtocolGuid, (VOID**)&Blk);
    if (EFI_ERROR (Status) || Blk == NULL) {
      continue;
    }

    Print (L"  BlockIo handle %u: MediaId=%u BlockSize=%u LastBlock=%llu MediaPresent=%u Removable=%u\n",
           (UINT32)i, (UINT32)Blk->Media->MediaId, (UINT32)Blk->Media->BlockSize,
           Blk->Media->LastBlock, (UINT32)Blk->Media->MediaPresent, (UINT32)Blk->Media->RemovableMedia);

    if (!Blk->Media->MediaPresent) {
      Print (L"    Media not present, skipping read.\n");
      continue;
    }

    UINTN BufSize = (UINTN)Blk->Media->BlockSize;
    VOID *Buf = AllocatePool (BufSize);
    if (Buf == NULL) {
      Print (L"    AllocatePool failed for %u bytes\n", (UINT32)BufSize);
      continue;
    }
    ZeroMem (Buf, BufSize);

    Status = Blk->ReadBlocks (Blk, Blk->Media->MediaId, 0, BufSize, Buf);
    Print (L"    ReadBlocks(LBA0) returned 0x%lx\n", Status);
    if (!EFI_ERROR (Status)) {
      DumpHex (L"    LBA0 (first 64 bytes)", (UINT8*)Buf, (BufSize > 64) ? 64 : BufSize);
      // Check for MBR signature 0x55AA at offset 0x1FE if block is >= 512
      if (BufSize >= 512) {
        UINT8 *b = (UINT8*)Buf;
        if (b[510] == 0x55 && b[511] == 0xAA) {
          Print (L"    MBR signature 0x55AA found at offset 0x1FE\n");
        } else {
          Print (L"    No MBR signature at 0x1FE (bytes=0x%02x 0x%02x)\n", b[510], b[511]);
        }
      }
    }
    FreePool (Buf);
  }

  if (Handles) {
    FreePool (Handles);
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
      // NOTE: some implementations allocate descriptor internally and return pointer; previous version used pointer.
      // Here we use stack-local struct version (UsbGetConfigDescriptor returns a pointer in original API; keep minimal)
      EFI_USB_CONFIG_DESCRIPTOR ConfigDesc;
      ZeroMem (&ConfigDesc, sizeof(ConfigDesc));
      Status = UsbIo->UsbGetConfigDescriptor (UsbIo, &ConfigDesc);
      PrintStatus (L"UsbGetConfigDescriptor", Status);
      if (!EFI_ERROR (Status)) {
        Print (L"  wTotalLength=%u bNumInterfaces=%u bConfigurationValue=%u\n",
               ConfigDesc.TotalLength, ConfigDesc.NumInterfaces, ConfigDesc.ConfigurationValue);
        UINTN DumpLen = (ConfigDesc.TotalLength > 64) ? 64 : ConfigDesc.TotalLength;
        Print(L" skipping DumpHex %u \n", DumpLen);
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


	// skipped this part - I have an errors here, and its freeze program
	// its returns names like Mass storage, or generic etc...

	/*
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
	*/

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
      PrintStatus (L"UsbPortReset (note: this may re-enumerate device)", Status);
    }

    Print (L"--- End of device %u checks ---\n", (UINT32)Index);
  }

  if (Handles != NULL) {
    FreePool (Handles);
  }
}

// few anothers helpers
STATIC
CONST CHAR16 *
KnownProtocolName (
  IN EFI_GUID *ProtocolGuid
  )
{
  // Compare common GUIDs you care about and return readable names.
  // Add more mappings if needed.
  if (CompareGuid (ProtocolGuid, &gEfiUsbIoProtocolGuid))            return L"EFI_USB_IO_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiBlockIoProtocolGuid))          return L"EFI_BLOCK_IO_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiSimpleFileSystemProtocolGuid)) return L"EFI_SIMPLE_FILE_SYSTEM_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiSimplePointerProtocolGuid))    return L"EFI_SIMPLE_POINTER_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiSimpleTextInProtocolGuid))     return L"EFI_SIMPLE_TEXT_IN_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiUsb2HcProtocolGuid))           return L"EFI_USB2_HC_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiUsbHcProtocolGuid))            return L"EFI_USB_HC_PROTOCOL";
  if (CompareGuid (ProtocolGuid, &gEfiPciIoProtocolGuid))            return L"EFI_PCI_IO_PROTOCOL";
  // fallback
  return NULL;
}

STATIC
VOID
ListHandlesAndProtocols (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN HandleCount = 0;
  UINTN i;

  Print (L"--- Dump handles (DevicePath) and installed protocols ---\n");

  // Locate handles that have a DevicePath (most device handles do)
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiDevicePathProtocolGuid, NULL, &HandleCount, &Handles);
  if (EFI_ERROR (Status) || HandleCount == 0) {
    Print (L"  LocateHandleBuffer(DevicePath) failed or none: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < HandleCount; ++i) {
    EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
    CHAR16 *DpStr = NULL;
    EFI_GUID **ProtocolArray = NULL;   // correct type: array of CONST EFI_GUID*
    UINTN ProtocolCount = 0;
    UINTN j;

	//EFI_DEVICE_PATH_PROTOCOL *DevPath;
	//CHAR16 *DpStr;
	Status = gBS->HandleProtocol (Handles[i], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
	if (!EFI_ERROR (Status) && DevPath != NULL) {
	  DpStr = MyDevicePathToStr (DevPath);
	  if (DpStr != NULL) {
		Print(L"DevicePath=%s\n", DpStr);
		FreePool(DpStr);
	  }
	}


	    //
    // SAFE: do not read DevicePath nodes here (suppressed). Just list protocols
    // without dereferencing GUID pointers (which may be invalid on some firmwares).
    //
    Print (L"Handle %u: DevicePath=(suppressed for stability)\n", (UINT32)i);

	//CONST EFI_GUID **ProtocolArray = NULL;
	//UINTN ProtocolCount = 0;
	//UINTN j;

	Status = gBS->ProtocolsPerHandle(Handles[i], (EFI_GUID ***)&ProtocolArray, &ProtocolCount);
	if (!EFI_ERROR(Status) && ProtocolArray != NULL) {
	  if (ProtocolCount > 1024) ProtocolCount = 1024; // safety cap
	  for (j = 0; j < ProtocolCount; ++j) {
		CONST EFI_GUID *g = ProtocolArray[j];
		if (g == NULL) {
		  Print(L"  [%u] NULL\n", (UINT32)j);
		  continue;
		}
		if (g == &gEfiUsbIoProtocolGuid) {
		  Print(L"  [%u] EFI_USB_IO_PROTOCOL\n", (UINT32)j);
		} else {
		  Print(L"  [%u] Protocol ptr=%p\n", (UINT32)j, (VOID*)g);
		}
	  }
	  FreePool((VOID*)ProtocolArray);
	}



	/*
    // ProtocolsPerHandle returns CONST EFI_GUID **ProtocolArray (allocated by the firmware)
    Status = gBS->ProtocolsPerHandle (Handles[i], (EFI_GUID ***)&ProtocolArray, &ProtocolCount);
    if (EFI_ERROR (Status)) {
      Print (L"  ProtocolsPerHandle failed: 0x%lx\n", Status);
      continue;
    }

    // Sanity cap for protocol count to avoid runaway values from buggy firmware
    if (ProtocolCount > 1024) {
      Print (L"  ProtocolCount suspicious (%u). Capping to 1024 for safety.\n", (UINT32)ProtocolCount);
      ProtocolCount = 1024;
    }

    for (j = 0; j < ProtocolCount; ++j) {
      EFI_GUID *GuidPtr = ProtocolArray[j];

      if (GuidPtr == NULL) {
        Print (L"    [%03u] <NULL GUID pointer>\n", (UINT32)j);
        continue;
      }

      //
      // Safe name detection: compare pointer values to known GUID variables.
      // This does NOT dereference the GUID at GuidPtr.
      //
      if (GuidPtr == &gEfiUsbIoProtocolGuid) {
        Print (L"    [%03u] EFI_USB_IO_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiBlockIoProtocolGuid) {
        Print (L"    [%03u] EFI_BLOCK_IO_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiSimpleFileSystemProtocolGuid) {
        Print (L"    [%03u] EFI_SIMPLE_FILE_SYSTEM_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiSimplePointerProtocolGuid) {
        Print (L"    [%03u] EFI_SIMPLE_POINTER_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiSimpleTextInProtocolGuid) {
        Print (L"    [%03u] EFI_SIMPLE_TEXT_IN_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiUsb2HcProtocolGuid) {
        Print (L"    [%03u] EFI_USB2_HC_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiUsbHcProtocolGuid) {
        Print (L"    [%03u] EFI_USB_HC_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else if (GuidPtr == &gEfiPciIoProtocolGuid) {
        Print (L"    [%03u] EFI_PCI_IO_PROTOCOL (ptr=%016lx)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      } else {
        // Unknown protocol: print the pointer value instead of dereferencing contents
        Print (L"    [%03u] Protocol ptr=%016lx (unknown)\n", (UINT32)j, (UINT64)(UINTN)GuidPtr);
      }
    }
	*/

    // Free the allocated ProtocolArray buffer (allocated by ProtocolsPerHandle)
    if (ProtocolArray != NULL) {
      FreePool ((VOID*)ProtocolArray);
      ProtocolArray = NULL;
    }



	/*
	    //
    // DO NOT call HandleProtocol(..., &gEfiDevicePathProtocolGuid, ...)
    // and DO NOT call DevicePathToStr() or MyDevicePathToStr() here --
    // those were triggering exceptions on your platform. Instead
    // simply print a placeholder and continue listing the protocols.
    //
    Print (L"Handle %u: DevicePath=(suppressed for stability)\n", (UINT32)i);

    // Get list of protocols installed on this handle
    // ProtocolsPerHandle will allocate ProtocolArray for us.
    Status = gBS->ProtocolsPerHandle (Handles[i], (EFI_GUID ***)&ProtocolArray, &ProtocolCount);
    if (EFI_ERROR (Status)) {
      Print (L"  ProtocolsPerHandle failed: 0x%lx\n", Status);
      // Do not attempt to free any DevicePath string (we never allocated it).
      continue;
    }

    // Defensive printing: do not read beyond ProtocolCount; check pointers.
    for (j = 0; j < ProtocolCount; ++j) {
      EFI_GUID *Guid = ProtocolArray[j];
      if (Guid == NULL) {
        Print (L"    [%03u] <NULL GUID pointer>\n", (UINT32)j);
        continue;
      }

      CONST CHAR16 *Name = KnownProtocolName (Guid);
      if (Name != NULL) {
        Print (L"    [%03u] %s\n", (UINT32)j, Name);
      } else {
        // Print raw GUID safely
        Print (L"    [%03u] GUID=%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
               (UINT32)j,
               (UINT32)Guid->Data1,
               (UINT32)Guid->Data2,
               (UINT32)Guid->Data3,
               (UINT32)Guid->Data4[0], (UINT32)Guid->Data4[1],
               (UINT32)Guid->Data4[2], (UINT32)Guid->Data4[3],
               (UINT32)Guid->Data4[4], (UINT32)Guid->Data4[5],
               (UINT32)Guid->Data4[6], (UINT32)Guid->Data4[7]);
      }
    }

    // Free the protocol array buffer allocated by ProtocolsPerHandle
    if (ProtocolArray != NULL) {
      FreePool ((VOID*)ProtocolArray);
      ProtocolArray = NULL;
    }
	*/


	/*
    Status = gBS->HandleProtocol (Handles[i], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
    if (!EFI_ERROR (Status) && DevPath != NULL) {
      DpStr = DevicePathToStr (DevPath);     // returns CHAR16*
      if (DpStr == NULL) {
        Print (L"Handle %u: DevicePathToStr returned NULL\n", (UINT32)i);
      }
    }
	*/

    //Print (L"Handle %u: DevicePath=%s\n", (UINT32)i, DpStr ? DpStr : L"(no device path)");
	
	/*
    // Get list of protocols installed on this handle
    Status = gBS->ProtocolsPerHandle (Handles[i], &ProtocolArray, &ProtocolCount);
    if (EFI_ERROR (Status)) {
      Print (L"  ProtocolsPerHandle failed: 0x%lx\n", Status);
      if (DpStr) FreePool (DpStr);
      continue;
    }

    for (j = 0; j < ProtocolCount; ++j) {
      EFI_GUID *Guid = ProtocolArray[j];
      CONST CHAR16 *Name = KnownProtocolName (Guid);
      if (Name != NULL) {
        Print (L"    [%03u] %s\n", (UINT32)j, Name);
      } else {
        // Print raw GUID
        Print (L"    [%03u] GUID=%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
               (UINT32)j,
               Guid->Data1, Guid->Data2, Guid->Data3,
               Guid->Data4[0], Guid->Data4[1],
               Guid->Data4[2], Guid->Data4[3], Guid->Data4[4], Guid->Data4[5], Guid->Data4[6], Guid->Data4[7]);
      }
    }

    // Free the protocol array buffer allocated by ProtocolsPerHandle
    if (ProtocolArray) {
      FreePool ((VOID*)ProtocolArray);
      ProtocolArray = NULL;
    }
	*/

    if (DpStr) {
      FreePool (DpStr);
      DpStr = NULL;
    }
  }

  FreePool (Handles);
}

/*
STATIC
BOOLEAN
ParseUsbPortFromDevicePathStr (
  IN CHAR16 *DpStr,
  OUT UINTN *PortNumber  // first port field
  )
{
  // Simple search for "USB(" and parse "USB(n" where n is decimal.
  CHAR16 *p = DpStr;
  while (p != NULL && *p != L'\0') {
    p = StrStr (p, L"USB(");
    if (p == NULL) {
      break;
    }
    p += 4; // point at first char after "USB("
    if ((*p) >= L'0' && (*p) <= L'9') {
      UINTN val = 0;
      while ((*p) >= L'0' && (*p) <= L'9') {
        val = val * 10 + (UINTN)(*p - L'0');
        p++;
      }
      *PortNumber = val;
      return TRUE;
    }
  }
  return FALSE;
}
*/

STATIC
VOID
PrintPortScSummary (
  IN UINT32 PortSc,
  IN UINT32 PortIndex
  )
{
  UINTN CCS = (PortSc >> 0) & 1;
  UINTN CSC = (PortSc >> 1) & 1;
  UINTN PE  = (PortSc >> 2) & 1;
  UINTN PE_CHANGE = (PortSc >> 3) & 1;
  UINTN PortOwner = (PortSc >> 13) & 1; // spec-dependent bit pos; used as indicative
  // simplified speed decode (specs vary by controller)
  UINTN PortSpeed = (PortSc >> 26) & 3;

  Print (L"      PORT%u: PORTSC raw=0x%08x  (CCS=%u CSC=%u PE=%u PEchg=%u PO=%u Speed=%u)\n",
         (UINT32)PortIndex, PortSc, (UINT32)CCS, (UINT32)CSC, (UINT32)PE, (UINT32)PE_CHANGE, (UINT32)PortOwner, (UINT32)PortSpeed);
}

STATIC
VOID
ExamineUsbPortEnumerationAndCompanionOwnership (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN PciCount = 0;
  UINTN i;

  Print (L"--- Examining USB controllers: ports vs published EFI_USB_IO_PROTOCOL handles ---\n");

  // Count all UsbIo handles in system (global)
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbHandleCount = 0;
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbHandleCount, &UsbHandles);
  if (EFI_ERROR (Status)) {
    UsbHandleCount = 0;
  }
  Print (L"Total EFI_USB_IO_PROTOCOL handles in system: %u\n", (UINT32)UsbHandleCount);

  // Locate all PCI handles
  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR (Status)) {
    Print (L"LocateHandleBuffer(PciIo) failed: 0x%lx\n", Status);
    if (UsbHandles) FreePool (UsbHandles);
    return;
  }

  for (i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo;
    UINT32 Data32 = 0;
    UINTN Bus, Device, Function, Segment;

    Status = gBS->HandleProtocol (PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR (Status) || PciIo == NULL) {
      continue;
    }

	// extra test
	//DiagPciBarAndMmio (PciIo);

    PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);

    // Read class/subclass/progif from config dword at offset 0x08
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &Data32);
    if (EFI_ERROR (Status)) {
      continue;
    }
	
	// extra test
	//DiagPciBarAndMmio (PciIo);
	
    UINT8 ClassCode = (UINT8)((Data32 >> 24) & 0xFF);
    UINT8 SubClass  = (UINT8)((Data32 >> 16) & 0xFF);
    UINT8 ProgIf    = (UINT8)((Data32 >> 8) & 0xFF);

    if (!(ClassCode == 0x0C && SubClass == 0x03)) {
      // Not a USB host controller
      continue;
    }

    Print (L"USB controller at PCI %02x:%02x.%x  ProgIf=0x%02x\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function, ProgIf);

    // Probe BARs to find capability base (CAPLENGTH at offset 0x00 in MMIO)
    BOOLEAN FoundMmio = FALSE;
    for (UINT8 BarIndex = 0; BarIndex < MAX_BARS && !FoundMmio; ++BarIndex) {
      UINT8 CapLength = 0xFF;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLength);
      if (EFI_ERROR (Status)) {
        continue;
      }
      if (CapLength == 0xFF || CapLength == 0x00) {
        continue;
      }

      // Found likely capability region
      FoundMmio = TRUE;
      UINTN OpBase = (UINTN)CapLength;

      // Read USBCMD, USBSTS
      UINT32 UsbCmd = 0, UsbSts = 0, HcSParams = 0;
      PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
      PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x04, 1, &UsbSts); // sometimes HCSPARAMS is at cap offset 0x04
      // For safety also read HCSPARAMS from capability offset 0x04
      PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);

      Print (L"  BAR%u CAPLENGTH=0x%02x  USBCMD=0x%08x USBSTS=0x%08x HCSPARAMS(raw)=0x%08x\n",
             (UINT32)BarIndex, CapLength, UsbCmd, UsbSts, HcSParams);

      // Determine number of ports (specs vary; we use lower 4 bits as approximate)
      UINT32 NPorts = (HcSParams & 0x0000000F);
      if (NPorts == 0) {
        // Some controllers provide bits differently; fall back to safe max
        NPorts = MAX_PORTS_SAFE;
      }
      if (NPorts > MAX_PORTS_SAFE) {
        NPorts = MAX_PORTS_SAFE;
      }

      // count connected ports for this controller
      UINT32 ConnectedCount = 0;

      for (UINT32 PortIndex = 1; PortIndex <= NPorts; ++PortIndex) {
        // PORTSC offset in operational regs: 0x44 + 4*(port-1)
        UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
        UINT32 PortSc = 0;
        EFI_STATUS r = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
        if (EFI_ERROR (r)) {
          // stop if read fails (no more ports)
          break;
        }
        // Decode relevant bits
        UINTN CCS = (PortSc >> 0) & 1;
        UINTN PO  = (PortSc >> 13) & 1; // example PO bit location (often bit 13)
        UINTN PE  = (PortSc >> 2) & 1;
        if (CCS) {
          ConnectedCount++;
        }
        // Print port summary
        PrintPortScSummary (PortSc, PortIndex);

        if (CCS && PO) {
          Print (L"        -> Port%u connected but PO (Port Owner) set: companion controller likely owns this port.\n", (UINT32)PortIndex);
        }

        if (CCS && !PE) {
          Print (L"        -> Port%u connected but not enabled (PE=0). Enumeration may not have completed.\n", (UINT32)PortIndex);
        }
      } // per-port loop

      Print (L"  Summary for controller %02x:%02x.%x: ConnectedPorts=%u\n", (UINT32)Bus, (UINT32)Device, (UINT32)Function, ConnectedCount);

      // Compare to global count of UsbIo handles (coarse check)
      if (ConnectedCount > 0 && (UINT32)UsbHandleCount == 0) {
        Print (L"  Warning: %u physical port(s) show connected devices but there are 0 EFI_USB_IO_PROTOCOL handles in the system.\n", ConnectedCount);
        Print (L"           Likely causes: missing USB bus driver, companion controller owning ports (PO=1), or controller halted.\n");
      } else if ((UINT32)UsbHandleCount < ConnectedCount) {
        Print (L"  Notice: %u connected ports vs %u UsbIo device handles overall. Some devices may be un-enumerated.\n",
               ConnectedCount, (UINT32)UsbHandleCount);
      } else {
        Print (L"  Device-handle count (%u) >= connected ports (%u) -- enumeration seems to have succeeded globally.\n",
               (UINT32)UsbHandleCount, ConnectedCount);
      }

      // HC state diagnostics
      // USBCMD bit 0 = Run/Stop (1=run), USBSTS bit 12 = HCHalted (1=halted) (bit positions may vary by spec)
      UINTN RunStop = UsbCmd & 1;
      UINTN HCHalted = (UsbSts >> 12) & 1;
      if (!RunStop) {
        Print (L"  Host Controller appears stopped (USBCMD.Run/Stop = 0). The HC driver may not have started the schedule.\n");
      }
      if (HCHalted) {
        Print (L"  Host Controller reports halted (USBSTS.HCHalted = 1). HC reset/driver issues may exist.\n");
      }

      // done with this BAR / controller
    } // per BAR

    if (!FoundMmio) {
      Print (L"  No accessible MMIO BAR found for this controller (skipped).\n");
    }
  } // per PCI handle

  if (PciHandles) FreePool (PciHandles);
  if (UsbHandles) FreePool (UsbHandles);

  Print (L"--- End of port enumeration vs handles check ---\n");
}


// /////////////////////// VERSION 1 //////////////////////////////////
// Defensive diagnostic: when PORT shows CCS=1 (connected), PE=0 (not enabled),
// and PO=0 (not owned by companion), call a deeper diagnostic function to inspect
// controller internal registers and schedule state.
//
// Usage: call CheckUnenumeratedPortsAndDiagnose() from UefiMain.
//
// Requires:
//   <Uefi.h>, <Library/UefiLib.h>, <Library/UefiBootServicesTableLib.h>,
//   <Protocol/PciIo.h>
//


/*
STATIC
VOID
DiagnosePortOnController (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               OpBase,
  IN UINT32              PortIndex
  )
{
  if (PciIo == NULL) {
    Print (L"  DiagnosePortOnController: PciIo is NULL\n");
    return;
  }

  EFI_STATUS Status;
  UINT32 UsbCmd = 0;
  UINT32 UsbSts = 0;
  UINT32 FrIndex = 0;
  UINT32 AsyncList = 0;
  UINT32 PeriodicBase = 0;
  UINT32 HcSParams = 0;
  UINT32 PortSc = 0;

  Print (L"  --- Diagnose: controller BAR%u OpBase=0x%03x port=%u ---\n", (UINT32)BarIndex, (UINT32)OpBase, (UINT32)PortIndex);

  // Read USBCMD (operational offset 0x00)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
  if (EFI_ERROR (Status)) {
    Print (L"    Read USBCMD failed: 0x%lx\n", Status);
    return;
  }

  // Read USBSTS (operational offset 0x04)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x04, 1, &UsbSts);
  if (EFI_ERROR (Status)) {
    Print (L"    Read USBSTS failed: 0x%lx\n", Status);
    return;
  }

  // Read FRINDEX (op offset 0x0C)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x0C, 1, &FrIndex);
  if (EFI_ERROR (Status)) {
    Print (L"    Read FRINDEX failed: 0x%lx\n", Status);
    // continue - FRINDEX is useful but not mandatory
    FrIndex = 0xFFFFFFFF;
  }

  // Read ASYNCLISTADDR (op offset 0x18)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x18, 1, &AsyncList);
  if (EFI_ERROR (Status)) {
    Print (L"    Read ASYNCLISTADDR failed: 0x%lx\n", Status);
    AsyncList = 0;
  }

  // Read PERIODICLISTBASE (op offset 0x10)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x10, 1, &PeriodicBase);
  if (EFI_ERROR (Status)) {
    Print (L"    Read PERIODICLISTBASE failed: 0x%lx\n", Status);
    PeriodicBase = 0;
  }

  // Read HCSPARAMS (capability offset 0x04)
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);
  if (EFI_ERROR (Status)) {
    Print (L"    Read HCSPARAMS failed: 0x%lx\n", Status);
    HcSParams = 0;
  }

  // Read PORTSC for this port
  UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
  if (EFI_ERROR (Status)) {
    Print (L"    Read PORT%u failed: 0x%lx\n", PortIndex, Status);
    return;
  }

  // Print basic register state
  Print (L"    USBCMD=0x%08x USBSTS=0x%08x FRINDEX=0x%08x\n", UsbCmd, UsbSts, FrIndex);
  Print (L"    ASYNCLISTADDR=0x%08x PERIODICLISTBASE=0x%08x HCSPARAMS=0x%08x\n", AsyncList, PeriodicBase, HcSParams);
  PrintPortScSummary (PortSc, PortIndex);

  // Decode suspicious cases
  UINTN RunStop = UsbCmd & 1;              // USBCMD bit 0
  UINTN HCHalted = (UsbSts >> 12) & 1;     // USBSTS.HCHalted (bit 12)
  UINTN AsyncEnabled = (UsbCmd >> 5) & 1;  // USBCMD Async Schedule Enable (bit 5)
  UINTN PeriodicEnabled = (UsbCmd >> 4) & 1; // USBCMD Periodic Schedule Enable (bit 4)

  if (!RunStop) {
    Print (L"    NOTE: Host controller Run/Stop == 0 (controller stopped). Driver may not have started schedules.\n");
  }
  if (HCHalted) {
    Print (L"    WARNING: HCHalted == 1 (controller halted). Needs driver reset or investigation.\n");
  }
  if (AsyncList == 0) {
    Print (L"    NOTE: ASYNCLISTADDR == 0 (no async list programmed). That prevents transfers for devices using async schedules.\n");
  } else {
    Print (L"    ASYNCLISTADDR points to 0x%08x\n", AsyncList);
  }
  if (!AsyncEnabled) {
    Print (L"    NOTE: Async schedule not enabled (USBCMD.AsyncEnable=0).\n");
  }
  if (!PeriodicEnabled) {
    Print (L"    NOTE: Periodic schedule not enabled (USBCMD.PeriodicEnable=0).\n");
  }

  // FRINDEX sanity: poll a few times to see if it advances (schedules running)
  if (FrIndex != 0xFFFFFFFF) {
    UINT32 prev = FrIndex;
    UINT32 cur = prev;
    UINTN advanced = 0;
    for (UINTN k = 0; k < FRINDEX_POLL_COUNT; ++k) {
      // delay
      gBS->Stall (FRINDEX_POLL_DELAY_MS * 1000); // Stall expects microseconds
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x0C, 1, &cur);
      if (EFI_ERROR (Status)) {
        Print (L"    Read FRINDEX failed during poll: 0x%lx\n", Status);
        cur = prev;
        break;
      }
      if (cur != prev) {
        advanced = 1;
        break;
      }
    }
    if (advanced) {
      Print (L"    FRINDEX advanced -> schedule appears to be running.\n");
    } else {
      Print (L"    FRINDEX did not advance in %u polls -> schedule may be stalled.\n", FRINDEX_POLL_COUNT);
    }
  } else {
    Print (L"    FRINDEX not available; cannot test schedule activity.\n");
  }

  // Check PORTSC change bits for hints (CSC, PED, etc.)
  UINTN CSC = (PortSc >> 1) & 1;
  UINTN PE   = (PortSc >> 2) & 1;
  UINTN PO   = (PortSc >> 13) & 1;
  UINTN CCS  = (PortSc >> 0) & 1;

  if (CCS && !PE && !PO) {
    Print (L"    Diagnosis: Connected but not enabled and EHCI owns the port (PO=0) -> likely HC driver or scheduling problem.\n");
    Print (L"    Suggested next steps: check HC driver presence, ensure Async list programmed and Async schedule enabled; enable debug logs in bus/HC driver.\n");
  }
  
  if (!CSC) {
	  Print(L" CSC is here but is not read anywhere \n ");
  }

  // Read a second snapshot of PORTSC to compare (non-destructive)
  UINT32 PortSc2 = 0;
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc2);
  if (!EFI_ERROR (Status)) {
    if (PortSc2 != PortSc) {
      Print (L"    PORTSC changed during diagnosis: old=0x%08x new=0x%08x\n", PortSc, PortSc2);
    } else {
      Print (L"    PORTSC unchanged during diagnosis.\n");
    }
  } else {
    Print (L"    Failed to re-read PORTSC: 0x%lx\n", Status);
  }

  Print (L"  --- End diagnose for port %u ---\n", PortIndex);
}
*/

// /////////////////////// VERSION 2 //////////////////////////////////
// Enhanced and defensive DiagnosePortOnController that reads CSC / change bits,
// checks for HC protocol presence on same handle, and polls PORTSC for changes.
//
// Requirements: <Uefi.h>, <Library/UefiLib.h>, <Library/UefiBootServicesTableLib.h>,
//               <Protocol/PciIo.h>, <Protocol/Usb2HostController.h>, <Protocol/UsbHostController.h>
//

#define PORTSC_POLL_CHANGE_COUNT 5
#define PORTSC_POLL_CHANGE_DELAY_US 200000  // 200 ms

/**
  Try to find a handle that exposes either gEfiUsb2HcProtocolGuid or gEfiUsbHcProtocolGuid
  which is bound to the same PCI device as the given PciIo (compare Segment:Bus:Dev:Func).
  Returns the matching handle or NULL if none found.

  This function does not allocate persistent state; caller must not free returned handle.
**/
STATIC
EFI_HANDLE
GetControllerHandleFromPciIo (
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
  EFI_STATUS Status;
  UINTN Bus, Device, Function, Segment;
  EFI_HANDLE *HcHandles = NULL;
  UINTN HcCount = 0;
  EFI_HANDLE ResultHandle = NULL;

  if (PciIo == NULL) {
    return NULL;
  }

  // get PCI location for comparison
  PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);

  //
  // Helper lambda-like: scan handles for a protocol and match pci location
  //
  #define TRY_FIND_PROTOCOL(protoGuid) do { \
    HcHandles = NULL; \
    HcCount = 0; \
    Status = gBS->LocateHandleBuffer (ByProtocol, (protoGuid), NULL, &HcCount, &HcHandles); \
    if (!EFI_ERROR (Status) && HcCount > 0) { \
      for (UINTN _h = 0; _h < HcCount; ++_h) { \
        EFI_PCI_IO_PROTOCOL *TmpPciIo = NULL; \
        if (!EFI_ERROR (gBS->HandleProtocol (HcHandles[_h], &gEfiPciIoProtocolGuid, (VOID**)&TmpPciIo)) && TmpPciIo != NULL) { \
          UINTN s2,b2,d2,f2; \
          TmpPciIo->GetLocation (TmpPciIo, &s2, &b2, &d2, &f2); \
          if (s2 == Segment && b2 == Bus && d2 == Device && f2 == Function) { \
            ResultHandle = HcHandles[_h]; \
            break; \
          } \
        } \
      } \
    } \
    if (HcHandles) { FreePool (HcHandles); HcHandles = NULL; HcCount = 0; } \
    if (ResultHandle != NULL) break; \
  } while (0)

  // Try USB2_HC (EHCI/xHCI-style)
  TRY_FIND_PROTOCOL (&gEfiUsb2HcProtocolGuid);

  // Try USB_HC (OHCI/UHCI legacy)
  if (ResultHandle == NULL) {
    TRY_FIND_PROTOCOL (&gEfiUsbHcProtocolGuid);
  }

  #undef TRY_FIND_PROTOCOL

  return ResultHandle; // NULL if not found
}


STATIC
VOID
DiagnosePortOnController (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               OpBase,
  IN UINT32              PortIndex,
  IN EFI_HANDLE          ControllerHandleOptional  // pass 0 if not known
  )
{
  if (PciIo == NULL) {
    Print (L"  DiagnosePortOnController: PciIo is NULL\n");
    return;
  }

  EFI_STATUS Status;
  UINT32 UsbCmd = 0, UsbSts = 0, FrIndex = 0;
  UINT32 AsyncList = 0, PeriodicBase = 0, HcSParams = 0;
  UINT32 PortSc = 0;

  Print (L"  --- Diagnose: controller BAR%u OpBase=0x%03x port=%u ---\n", (UINT32)BarIndex, (UINT32)OpBase, (UINT32)PortIndex);

  // Read main op regs
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
  if (EFI_ERROR (Status)) {
    Print (L"    Read USBCMD failed: 0x%lx\n", Status);
    return;
  }
  PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x04, 1, &UsbSts);
  PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x0C, 1, &FrIndex);
  PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x18, 1, &AsyncList);
  PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x10, 1, &PeriodicBase);
  PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);

  // Read PORTSC
  UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
  if (EFI_ERROR (Status)) {
    Print (L"    Read PORT%u failed: 0x%lx\n", (UINT32)PortIndex, Status);
    return;
  }

  // print basic values
  Print (L"    USBCMD=0x%08x USBSTS=0x%08x FRINDEX=0x%08x\n", UsbCmd, UsbSts, FrIndex);
  Print (L"    ASYNCLISTADDR=0x%08x PERIODICLISTBASE=0x%08x HCSPARAMS=0x%08x\n", AsyncList, PeriodicBase, HcSParams);
  PrintPortScSummary (PortSc, PortIndex);

  // Decode change bits precisely (EHCI spec typical positions):
  // CCS (bit0), CSC (bit1), PE (bit2), PED (bit3), CPE (bit12), PO (bit13)
  UINTN CCS = (PortSc >> 0) & 1;
  UINTN CSC = (PortSc >> 1) & 1;
  UINTN PE  = (PortSc >> 2) & 1;
  UINTN PED = (PortSc >> 3) & 1;
  UINTN CPE = (PortSc >> 12) & 1;  // Connect Port Change? (spec varies)
  UINTN PO  = (PortSc >> 13) & 1;

  if (CSC) {
    Print (L"    PORT%u: Connect Status Change (CSC) = 1 -> a connect/disconnect event occurred since last clear.\n", (UINT32)PortIndex);
  }
  if (PED) {
    Print (L"    PORT%u: Port Enable/Disable Change (PED) = 1 -> port enable state changed.\n", (UINT32)PortIndex);
  }
  if (CPE) {
    Print (L"    PORT%u: PortEnableChange (CPE) = 1 -> endpoint enable state changed.\n", (UINT32)PortIndex);
  }

  // HC-level hints
  UINTN RunStop = UsbCmd & 1;
  UINTN HCHalted = (UsbSts >> 12) & 1;
  UINTN AsyncEnabled = (UsbCmd >> 5) & 1;
  UINTN PeriodicEnabled = (UsbCmd >> 4) & 1;

  if (!RunStop) {
    Print (L"    NOTE: Host controller Run/Stop == 0 (controller stopped). Driver may not have started schedules.\n");
  }
  if (HCHalted) {
    Print (L"    WARNING: HCHalted == 1 (controller halted). Needs HC driver attention.\n");
  }
  if (AsyncList == 0) {
    Print (L"    NOTE: ASYNCLISTADDR == 0 (no async list programmed).\n");
  }
  if (!AsyncEnabled) {
    Print (L"    NOTE: Async schedule not enabled (USBCMD.AsyncEnable=0).\n");
  }
  if (!PeriodicEnabled) {
    Print (L"    NOTE: Periodic schedule not enabled (USBCMD.PeriodicEnable=0).\n");
  }

  // Best-effort: check whether host controller protocol is published on same handle.
  // This is harmless: try HandleProtocol for USB2_HC / USB_HC.
  {
    EFI_HANDLE CandidateHandle = ControllerHandleOptional;
    BOOLEAN HcProtocolFound = FALSE;

    // If caller didn't give a handle, attempt to locate PciIo's handle by walking handles that expose PciIo.
    // Here we try to find matching handle by comparing Pci location (safe best-effort).
    if (CandidateHandle == NULL) {
      // try to recover handle: call PciIo->GetLocation to get B:D.F and then locate matching handle
      UINTN Bus, Device, Function, Segment;
      PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);
      // Try to find a handle that exposes Usb2Hc protocol and also PciIo with same location.
      EFI_HANDLE *HcHandles = NULL;
      UINTN HcCount = 0;
      if (!EFI_ERROR (gBS->LocateHandleBuffer (ByProtocol, &gEfiUsb2HcProtocolGuid, NULL, &HcCount, &HcHandles)) && HcCount > 0) {
        for (UINTN hi = 0; hi < HcCount; ++hi) {
          EFI_PCI_IO_PROTOCOL *TmpPciIo = NULL;
          if (!EFI_ERROR (gBS->HandleProtocol (HcHandles[hi], &gEfiPciIoProtocolGuid, (VOID**)&TmpPciIo)) && TmpPciIo != NULL) {
            UINTN b,d,f,s;
            TmpPciIo->GetLocation (TmpPciIo, &s, &b, &d, &f);
            if (b == Bus && d == Device && f == Function && s == Segment) {
              CandidateHandle = HcHandles[hi];
              break;
            }
          }
        }
        FreePool (HcHandles);
      }
    }

    if (CandidateHandle != NULL) {
      // Try USB2_HC
      VOID *Proto = NULL;
      if (!EFI_ERROR (gBS->HandleProtocol (CandidateHandle, &gEfiUsb2HcProtocolGuid, &Proto))) {
        Print (L"    Found EFI_USB2_HC_PROTOCOL on same handle -> EHCI driver published.\n");
        HcProtocolFound = TRUE;
      } else if (!EFI_ERROR (gBS->HandleProtocol (CandidateHandle, &gEfiUsbHcProtocolGuid, &Proto))) {
        Print (L"    Found EFI_USB_HC_PROTOCOL (OHCI/UHCI) on same handle.\n");
        HcProtocolFound = TRUE;
      } else {
        Print (L"    No USB2_HC/USB_HC protocol found on candidate handle (maybe driver installed elsewhere).\n");
      }
    } else {
      Print (L"    Could not determine HC protocol handle by location (best-effort failed).\n");
    }

    if (!HcProtocolFound) {
      Print (L"    Suggestion: ensure HC DXE driver (EHCI/OHCI/UHCI) for this chipset is present in your firmware build.\n");
    }
  }

  // Poll PORTSC a few times to see if change bits appear (do not clear them)
  {
    UINT32 initial = PortSc;
    for (UINTN p = 0; p < PORTSC_POLL_CHANGE_COUNT; ++p) {
      gBS->Stall (PORTSC_POLL_CHANGE_DELAY_US); // microseconds
      UINT32 newPortSc = 0;
      EFI_STATUS rr = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &newPortSc);
      if (EFI_ERROR (rr)) {
        Print (L"    Re-read PORTSC failed: 0x%lx\n", rr);
        break;
      }
      if (newPortSc != initial) {
        Print (L"    PORTSC changed during polling: old=0x%08x new=0x%08x\n", initial, newPortSc);
        // decode if change bits set now
        if (((newPortSc >> 1) & 1) != ((initial >> 1) & 1)) {
          Print (L"      CSC toggled\n");
        }
        if (((newPortSc >> 3) & 1) != ((initial >> 3) & 1)) {
          Print (L"      PED toggled\n");
        }
        initial = newPortSc;
      }
    } // poll loop
  }

  // final advice
  if (CCS && !PE && !PO) {
    Print (L"    Result: connected, not enabled, EHCI owns port -> likely HC scheduling/driver not started.\n");
    Print (L"    Next safe diagnostic steps: (1) verify HC driver is present in DXE image; (2) enable driver debug logging; (3) check for companion controller and driver.\n");
    Print (L"    If you control firmware build: ensure UsbBusDxe and relevant host-controller drivers are included and started.\n");
  }

  Print (L"  --- End diagnose for port %u ---\n", PortIndex);
}


/////////////////////////////////
STATIC
VOID
TryConnectController (
  IN EFI_HANDLE ControllerHandle
  )
{
  EFI_STATUS Status;

  if (ControllerHandle == NULL) {
    Print (L"TryConnectController: invalid ControllerHandle (NULL)\n");
    return;
  }

  Print (L"Trying gBS->ConnectController for handle %p ...\n", ControllerHandle);
  Status = gBS->ConnectController (ControllerHandle, NULL, NULL, TRUE);
  if (EFI_ERROR (Status)) {
    Print (L"  ConnectController failed: 0x%lx\n", Status);
  } else {
    Print (L"  ConnectController returned SUCCESS (driver bind/start requested)\n");
  }
}


STATIC
VOID
TryStartSpecificDriverForController (
  IN EFI_HANDLE ControllerHandle
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *DriverHandles = NULL;
  UINTN Count = 0;
  UINTN i;

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiDriverBindingProtocolGuid, NULL, &Count, &DriverHandles);
  if (EFI_ERROR(Status) || Count == 0) {
    Print (L"  No DriverBinding handles found: 0x%lx\n", Status);
    return;
  }

  for (i = 0; i < Count; ++i) {
    EFI_DRIVER_BINDING_PROTOCOL *DrvBinding = NULL;

    Status = gBS->HandleProtocol (DriverHandles[i], &gEfiDriverBindingProtocolGuid, (VOID**)&DrvBinding);
    if (EFI_ERROR(Status) || DrvBinding == NULL) {
      continue;
    }

    // Ask if this driver *supports* our controller
    Status = DrvBinding->Supported (DrvBinding, ControllerHandle, NULL);
    if (Status == EFI_SUCCESS) {
      Print (L"  DriverBinding handle %p reports Supported -> calling Start()\n", DriverHandles[i]);
      Status = DrvBinding->Start (DrvBinding, ControllerHandle, NULL);
      Print (L"    Start returned 0x%lx\n", Status);
      // optionally break after first success
    }
  }

  if (DriverHandles) FreePool (DriverHandles);
}
//////////////////////

STATIC
VOID
TryStartDriversForController (
  IN EFI_HANDLE ControllerHandle
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *DriverHandles = NULL;
  UINTN Count = 0;
  UINTN i;

  if (ControllerHandle == NULL) {
    Print(L"TryStartDriversForController: ControllerHandle is NULL\n");
    return;
  }

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiDriverBindingProtocolGuid, NULL, &Count, &DriverHandles);
  if (EFI_ERROR (Status) || Count == 0) {
    Print (L"  No DriverBinding handles found: 0x%lx\n", Status);
    return;
  }

  Print (L"  Trying DriverBinding.Start() on %u driver-binding handles...\n", (UINT32)Count);

  for (i = 0; i < Count; ++i) {
    EFI_DRIVER_BINDING_PROTOCOL *DrvBinding = NULL;

    Status = gBS->HandleProtocol (DriverHandles[i], &gEfiDriverBindingProtocolGuid, (VOID**)&DrvBinding);
    if (EFI_ERROR (Status) || DrvBinding == NULL) {
      continue;
    }

    // Query Supported() — pass NULL for RemainingDevicePath (typical).
    Status = DrvBinding->Supported (DrvBinding, ControllerHandle, NULL);

    // Print the Supported() returned code so you can see why none matched.
    Print (L"    DriverBinding handle %p Supported() => 0x%lx\n", DriverHandles[i], Status);

    if (Status == EFI_SUCCESS) {
      // This driver says it supports our controller — try to Start it.
      Print (L"      -> Supported; calling Start() now...\n");
      Status = DrvBinding->Start (DrvBinding, ControllerHandle, NULL);
      Print (L"      Start() returned 0x%lx\n", Status);
      // Don't break: optionally try other drivers too to collect more diagnostics.
    }
  }

  if (DriverHandles) {
    FreePool (DriverHandles);
  }
}


STATIC
EFI_STATUS
TryForceControllerRun (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               OpBase
  )
{
  EFI_STATUS Status;
  UINT32 UsbCmd = 0;
  UINT32 ReadBack = 0;

  if (PciIo == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
  if (EFI_ERROR (Status)) {
    Print (L"    TryForceControllerRun: read USBCMD failed: 0x%lx\n", Status);
    return Status;
  }

  Print (L"    Current USBCMD=0x%08x -> attempting to set Run+Periodic+Async bits (experimental)\n", UsbCmd);

  // Set Run (bit 0), Periodic Enable (bit 4), Async Enable (bit 5)
  UsbCmd |= (1u << 0) | (1u << 4) | (1u << 5);

  Status = PciIo->Mem.Write (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
  if (EFI_ERROR (Status)) {
    Print (L"    TryForceControllerRun: write USBCMD failed: 0x%lx\n", Status);
    return Status;
  }

  // optional read-back verification
  Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &ReadBack);
  if (!EFI_ERROR (Status)) {
    Print (L"    Wrote USBCMD -> ReadBack USBCMD=0x%08x\n", ReadBack);
  } else {
    Print (L"    Wrote USBCMD but read-back failed: 0x%lx\n", Status);
  }

  // small delay to let HW react
  gBS->Stall (100 * 1000); // 100 ms

  return EFI_SUCCESS;
}

STATIC
VOID
PrintPciVendorDevice (
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
  EFI_STATUS Status;
  UINT32 Vd = 0;
  if (PciIo == NULL) return;
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x00, 1, &Vd);
  if (!EFI_ERROR (Status)) {
    Print (L"    PCI Vendor/Device: 0x%08x (Vendor=0x%04x Device=0x%04x)\n", Vd, (UINT32)(Vd & 0xffff), (UINT32)((Vd >> 16) & 0xffff));
  } else {
    Print (L"    Failed to read PCI Vendor/Device: 0x%lx\n", Status);
  }
}


///////////////////////////

STATIC
VOID
CheckUnenumeratedPortsAndDiagnose (
  VOID
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN HandleCount = 0;

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &HandleCount, &PciHandles);
  if (EFI_ERROR (Status)) {
    Print (L"CheckUnenumeratedPortsAndDiagnose: LocateHandleBuffer(PciIo) failed: 0x%lx\n", Status);
    return;
  }

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol (PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR (Status) || PciIo == NULL) {
      continue;
    }
	
	// extra test
	//DiagPciBarAndMmio (PciIo);

    // Read class/subclass/progif to ensure it's a USB host controller
    UINT32 ClassDword = 0;
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassDword);
    if (EFI_ERROR (Status)) {
      continue;
    }
    UINT8 ClassCode = (UINT8)((ClassDword >> 24) & 0xFF);
    UINT8 SubClass  = (UINT8)((ClassDword >> 16) & 0xFF);
    if (!(ClassCode == 0x0C && SubClass == 0x03)) {
      continue; // skip non-USB controllers
    }

    // Probe BARs to locate capability/op region
    for (UINT8 BarIndex = 0; BarIndex < MAX_BARS; ++BarIndex) {
      UINT8 CapLen = 0xFF;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLen);
      if (EFI_ERROR (Status)) {
        continue;
      }
      if (CapLen == 0xFF || CapLen == 0x00) {
        continue;
      }

      UINTN OpBase = (UINTN)CapLen;
      // Read HCSPARAMS from capability offset 0x04
      UINT32 HcSParams = 0;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);
      if (EFI_ERROR (Status)) {
        // fallback: assume some ports but continue defensively
        HcSParams = 0;
      }

      UINT32 NPorts = (HcSParams & 0x0000000F);
      if (NPorts == 0) {
        NPorts = MAX_PORTS_SAFE;
      }
      if (NPorts > MAX_PORTS_SAFE) {
        NPorts = MAX_PORTS_SAFE;
      }

      for (UINT32 PortIndex = 1; PortIndex <= NPorts; ++PortIndex) {
        UINT32 PortSc = 0;
        UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
        Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
        if (EFI_ERROR (Status)) {
          break; // stop per-port loop (no more ports or read error)
        }

        UINTN CCS = (PortSc >> 0) & 1;
        UINTN PE  = (PortSc >> 2) & 1;
        UINTN PO  = (PortSc >> 13) & 1; // approximate bit pos; ok for diagnostics

		/*
        if (CCS && !PE && !PO) {
          // Found candidate un-enumerated port (connected, not enabled, EHCI owns)
          Print (L"Controller PCI BAR%u: Found port %u with CCS=1 PE=0 PO=0 -> diagnosing deeper\n", (UINT32)BarIndex, (UINT32)PortIndex);
          DiagnosePortOnController (PciIo, (UINT8)BarIndex, OpBase, PortIndex, NULL); // cast on UINT8 is my change. This was not produce by GPT-5 !!!! 
          // continue checking other ports (don't break)
        }
		*/
		
		// inside the per-port detection loop, when CCS && !PE && !PO found:
		if (CCS && !PE && !PO) {
		  Print (L"Controller PCI BAR%u: Found port %u with CCS=1 PE=0 PO=0 -> diagnosing deeper\n", (UINT32)BarIndex, (UINT32)PortIndex);

		  // best-effort find HC protocol handle for better diagnostics
		  EFI_HANDLE HcHandle = GetControllerHandleFromPciIo (PciIo); // may return NULL
		  if (HcHandle != NULL) {
			Print (L"  Matched HC protocol handle: %p\n", HcHandle);
		  } else {
			Print (L"  No HC protocol handle found for this PCI device (driver may not have bound)\n");
		  }
		  
		  /*
		  gBS->Stall (200 * 1000);
		    // Try to ask DXE drivers to bind to this PCI handle
			 TryConnectController (PciHandles[i]); // you'll have that handle in your loop


			gBS->Stall (200 * 1000); // 200ms pause to let driver bind and start
		  // Call diagnosis, passing candidate handle (may be NULL)
		  DiagnosePortOnController (PciIo, (UINT8)BarIndex, OpBase, PortIndex, HcHandle);
		  gBS->Stall (200 * 1000); // 200ms pause to let driver bind and start
			*/
		  // continue checking other ports (don't break)
		  
		   // Print PCI vendor/device for triage
		  PrintPciVendorDevice (PciIo);

		  // 1) Try driver-binding Start() for any driver that reports Supported() for this controller
		  TryStartDriversForController (PciHandles[i]); // pass the controller handle from outer loop
		  // small pause
		  gBS->Stall (200 * 1000); // 200 ms

		  // 2) Try ConnectController as you already do (optional, harmless)
		  TryConnectController (PciHandles[i]);
		  gBS->Stall (200 * 1000); // give time for Start/Connect to complete

		  // 3) Re-run diagnostic to see if driver started schedules
		  DiagnosePortOnController (PciIo, (UINT8)BarIndex, OpBase, PortIndex, HcHandle);

		  // 4) If still stopped and you want to try HW-level start, call experimental force-run
		  //    WARNING: risky. Use only on test hardware.
		  {
			UINT32 UsbCmd = 0;
			// read USBCMD to see if changed
			PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
			if (UsbCmd == 0) {
			  Print (L"  USBCMD still 0 after attempting driver start. Optionally forcing controller run (experimental)...\n");
			  EFI_STATUS r = TryForceControllerRun (PciIo, (UINT8)BarIndex, OpBase);
			  Print (L"    TryForceControllerRun returned 0x%lx\n", r);
			  // Wait then re-diagnose
			  gBS->Stall (200 * 1000);
			  DiagnosePortOnController (PciIo, (UINT8)BarIndex, OpBase, PortIndex, HcHandle);
			} else {
			  Print (L"  USBCMD changed to 0x%08x — driver may have started. Re-diagnosing above.\n", UsbCmd);
			}
		  }
		  
		  
		}
		
      } // per port loop
    } // per BAR loop
  } // per PCI handle

  if (PciHandles) {
    FreePool (PciHandles);
  }
}


// Helper: check if a device path contains PCI node with given Device/Function bytes
STATIC
BOOLEAN
DevicePathContainsPciNodeMatch (
  IN CONST EFI_DEVICE_PATH_PROTOCOL *DevicePath,
  IN UINT8 TargetDevice,
  IN UINT8 TargetFunction
  )
{
  CONST EFI_DEVICE_PATH_PROTOCOL *Node = DevicePath;
  while (Node != NULL) {
    UINT8 Type = Node->Type;
    UINT8 SubType = Node->SubType;
    UINT16 NodeLen = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));
    if (Type == DP_TYPE_END && SubType == DP_SUBTYPE_END_ENTIRE) break;
    if (NodeLen < 4) break;

    if (Type == DP_TYPE_HARDWARE && SubType == DP_SUBTYPE_PCI && NodeLen >= 6) {
      UINT8 *raw = (UINT8*)Node;
      UINT8 Dev = raw[4];
      UINT8 Func = raw[5];
      if (Dev == TargetDevice && Func == TargetFunction) {
        return TRUE;
      }
    }
    Node = (CONST EFI_DEVICE_PATH_PROTOCOL *) ((CONST UINT8 *) Node + NodeLen);
  }
  return FALSE;
}

// Dump basic PCI + BARs and full first 0x40 bytes of PCI config
STATIC
VOID
DumpPciBasicAndConfig (
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
  EFI_STATUS Status;
  UINT32 Val32;
  UINT16 Cmd = 0;
  UINTN Segment, Bus, Device, Function;

  if (PciIo == NULL) return;
  PciIo->GetLocation (PciIo, &Segment, &Bus, &Device, &Function);
  Print (L"  -> PCI location: Segment=%u Bus=%02x Dev=%02x Func=%x\n", (UINT32)Segment, (UINT32)Bus, (UINT32)Device, (UINT32)Function);

  // Vendor/Device (offset 0x00)
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x00, 1, &Val32);
  if (!EFI_ERROR(Status)) {
    Print (L"    Vendor/Device = 0x%08x (Vendor=0x%04x Device=0x%04x)\n", Val32, (UINT32)(Val32 & 0xffff), (UINT32)((Val32 >> 16) & 0xffff));
  }

  // Command register (offset 0x04, 16-bit)
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint16, 0x04, 1, &Cmd);
  if (!EFI_ERROR(Status)) {
    Print (L"    PCI Command = 0x%04x (IO=%d MEM=%d BusMaster=%d)\n", Cmd, (Cmd & 1) ? 1 : 0, (Cmd & 2) ? 1 : 0, (Cmd & 4) ? 1 : 0);
  } else {
    Print (L"    Failed to read PCI Command: 0x%lx\n", Status);
  }

  // BAR0 at offset 0x10..0x13
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x10, 1, &Val32);
  if (!EFI_ERROR(Status)) {
    Print (L"    PCI BAR0 = 0x%08x\n", Val32);
  }

  // Dump full first 0x40 bytes of PCI config space for comparison
  Print (L"    PCI config (0x00..0x3C):\n");
  for (UINTN off = 0; off < 0x40; off += 4) {
    PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, (UINT32)off, 1, &Val32);
    Print (L"      [%02x] = 0x%08x\n", (UINT32)off, Val32);
  }
}

// Try to get hub descriptor via class request (if hub); safe IN transfer
#define USB_HUB_DESCRIPTOR_TYPE 0x29
STATIC
VOID
TryGetHubDescriptor (
  IN EFI_USB_IO_PROTOCOL *UsbIo
  )
{
  EFI_STATUS Status;
  EFI_USB_DEVICE_REQUEST Req;
  UINT8 Buf[8];
  UINT32 UsbStatus = 0;

  if (UsbIo == NULL) return;

  ZeroMem (&Req, sizeof(Req));
  Req.RequestType = USB_ENDPOINT_DIR_IN | USB_REQ_TYPE_CLASS | USB_TARGET_DEVICE;
  Req.Request = USB_REQ_GET_DESCRIPTOR; // 0x06
  Req.Value = (USB_HUB_DESCRIPTOR_TYPE << 8) | 0; // wValue: (dtype<<8) | index
  Req.Index = 0; // for hub device (device-level)
  Req.Length = sizeof(Buf);

  ZeroMem (Buf, sizeof(Buf));
  Status = UsbIo->UsbControlTransfer (UsbIo, &Req, EfiUsbDataIn, TIMEOUT_MS, Buf, sizeof(Buf), &UsbStatus);
  PrintStatus (L"  TryGetHubDescriptor (initial 8 bytes)", Status);
  if (EFI_ERROR(Status)) return;

  // First byte often bDescLength -> could be longer; print what we got
  Print (L"    HubDesc (first %u bytes):\n", (UINT32)sizeof(Buf));
  DumpHex (NULL, Buf, sizeof(Buf));
}

STATIC
VOID
ListAllUsbIoHandlesWithPortNodes (VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbCount = 0;

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
  if (EFI_ERROR(Status)) {
    Print (L"ListAllUsbIoHandlesWithPortNodes: LocateHandleBuffer(UsbIo) failed: 0x%lx\n", Status);
    return;
  }

  Print (L"--- Listing all %u EFI_USB_IO_PROTOCOL handles and USB() nodes ---\n", (UINT32)UsbCount);

  for (UINTN i = 0; i < UsbCount; ++i) {
    EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
    CHAR16 *DpStr = NULL;

    Print (L"UsbIo handle[%u] = %p\n", (UINT32)i, UsbHandles[i]);
    // Print DevicePath string if present
    Status = gBS->HandleProtocol (UsbHandles[i], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
    if (!EFI_ERROR(Status) && DevPath != NULL) {
      DpStr = MyDevicePathToStr (DevPath);
      if (DpStr) {
        Print (L"  DevicePath: %s\n", DpStr);
        FreePool (DpStr);
      }
      // Walk devicepath nodes and print USB(port,iface) nodes
      CONST EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;
      while (Node != NULL) {
        UINT8 Type = Node->Type;
        UINT8 SubType = Node->SubType;
        UINT16 NodeLen = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));
        if (Type == DP_TYPE_END && SubType == DP_SUBTYPE_END_ENTIRE) break;
        if (NodeLen < 4) break;
        if (Type == DP_TYPE_MESSAGING && SubType == DP_SUBTYPE_USB && NodeLen >= 6) {
          UINT8 *raw = (UINT8*)Node;
          UINT8 ParentPort = raw[4];
          UINT8 Interface = raw[5];
          Print (L"  -> USB node: ParentPort=%u Interface=%u\n", (UINT32)ParentPort, (UINT32)Interface);
        }
        Node = (CONST EFI_DEVICE_PATH_PROTOCOL *) ((CONST UINT8 *) Node + NodeLen);
      }
    } else {
      Print (L"  No DevicePath installed (or couldn't read DevicePath protocol)\n");
    }

    // Also print the device descriptor (if you want more details)
    {
      EFI_USB_IO_PROTOCOL *UsbIo = NULL;
      Status = gBS->HandleProtocol (UsbHandles[i], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
      if (!EFI_ERROR(Status) && UsbIo != NULL) {
        EFI_USB_DEVICE_DESCRIPTOR DevDesc;
        ZeroMem (&DevDesc, sizeof(DevDesc));
        Status = UsbIo->UsbGetDeviceDescriptor (UsbIo, &DevDesc);
        if (!EFI_ERROR(Status)) {
          Print (L"  DeviceDescriptor: Vendor=0x%04x Product=0x%04x bDeviceClass=0x%02x\n",
                 DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
        } else {
          Print (L"  UsbGetDeviceDescriptor failed: 0x%lx\n", Status);
        }
      }
    }
  }

  if (UsbHandles) FreePool (UsbHandles);
  Print (L"--- End list ---\n");
}

/*
STATIC
VOID
DumpDriverHandleAndTryBind (
  IN EFI_HANDLE ControllerHandle,
  IN EFI_HANDLE DriverBindingHandle
  )
{
  EFI_STATUS Status;

  if (ControllerHandle == NULL || DriverBindingHandle == NULL) {
    Print (L"DumpDriverHandleAndTryBind: invalid handle(s)\n");
    return;
  }

  Print (L"--- DumpDriverHandleAndTryBind => controller=%p driverBinding=%p ---\n",
         ControllerHandle, DriverBindingHandle);

  // 1) protocols present on driver handle (ProtocolsPerHandle)
  EFI_GUID **ProtocolArray = NULL;
  UINTN ProtocolCount = 0;
  Status = gBS->ProtocolsPerHandle (DriverBindingHandle, &ProtocolArray, &ProtocolCount);
  if (!EFI_ERROR(Status) && ProtocolArray != NULL) {
    Print (L"  Protocols on driver handle: %u\n", (UINT32)ProtocolCount);
    for (UINTN i = 0; i < ProtocolCount; ++i) {
      CONST EFI_GUID *g = ProtocolArray[i];
      if (g == NULL) {
        Print (L"    [%u] NULL\n", (UINT32)i);
      } else if (CompareGuid (g, &gEfiDriverBindingProtocolGuid)) {
        Print (L"    [%u] EFI_DRIVER_BINDING_PROTOCOL\n", (UINT32)i);
      } else if (CompareGuid (g, &gEfiLoadedImageProtocolGuid)) {
        Print (L"    [%u] EFI_LOADED_IMAGE_PROTOCOL\n", (UINT32)i);
      } else {
        // print pointer value; device path GUID printing isn't portable here
        Print (L"    [%u] GUID ptr=%p\n", (UINT32)i, (VOID*)g);
      }
    }
    FreePool (ProtocolArray);
  } else {
    Print (L"  ProtocolsPerHandle failed: 0x%lx\n", Status);
  }

  // 2) Try to get LoadedImage to find the image path (if present)
  EFI_LOADED_IMAGE_PROTOCOL *LoadedImage = NULL;
  Status = gBS->HandleProtocol (DriverBindingHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
  if (!EFI_ERROR(Status) && LoadedImage != NULL) {
    Print (L"  LoadedImage: ImageBase=%p FilePath=%p\n", LoadedImage->ImageBase, LoadedImage->FilePath);
    if (LoadedImage->FilePath) {
      CHAR16 *DpStr = MyDevicePathToStr (LoadedImage->FilePath);
      if (DpStr) {
        Print (L"    Driver image device path: %s\n", DpStr);
        FreePool (DpStr);
      }
    }
  } else {
    Print (L"  No LoadedImageProtocol on driver handle (status=0x%lx)\n", Status);
  }

  // 3) Try ConnectController with this driver handle explicitly
  Status = gBS->ConnectController (ControllerHandle, DriverBindingHandle, NULL, TRUE);
  Print (L"  ConnectController(controller=%p, driver=%p) returned 0x%lx\n", ControllerHandle, DriverBindingHandle, Status);

  Print (L"--- End dump/try ---\n");
}
*/

STATIC
VOID
DumpDriverHandleAndTryBind (
  IN EFI_HANDLE ControllerHandle,
  IN EFI_HANDLE DriverBindingHandle
  )
{
  EFI_STATUS Status;

  if (ControllerHandle == NULL || DriverBindingHandle == NULL) {
    Print (L"DumpDriverHandleAndTryBind: invalid handle(s)\n");
    return;
  }

  Print (L"--- DumpDriverHandleAndTryBind => controller=%p driverBinding=%p ---\n",
         ControllerHandle, DriverBindingHandle);

  //
  // 1) List protocols on the driver handle via ProtocolsPerHandle
  //
  EFI_GUID **ProtocolArray = NULL;
  UINTN ProtocolCount = 0;

  Status = gBS->ProtocolsPerHandle (DriverBindingHandle, &ProtocolArray, &ProtocolCount);
  if (!EFI_ERROR (Status) && ProtocolArray != NULL) {
    Print (L"  Protocols on driver handle: %u\n", (UINT32)ProtocolCount);
    for (UINTN i = 0; i < ProtocolCount; ++i) {
      CONST EFI_GUID *g = ProtocolArray[i];
      if (g == NULL) {
        Print (L"    [%u] NULL\n", (UINT32)i);
      } else if (CompareGuid (g, &gEfiDriverBindingProtocolGuid)) {
        Print (L"    [%u] EFI_DRIVER_BINDING_PROTOCOL\n", (UINT32)i);
      } else if (CompareGuid (g, &gEfiLoadedImageProtocolGuid)) {
        Print (L"    [%u] EFI_LOADED_IMAGE_PROTOCOL\n", (UINT32)i);
      } else {
        // Print pointer value only (printing GUID as %g is not portable here)
        Print (L"    [%u] GUID ptr=%p\n", (UINT32)i, (VOID*)g);
      }
    }
    // ProtocolsPerHandle allocates the buffer; free it.
    FreePool (ProtocolArray);
    ProtocolArray = NULL;
  } else {
    Print (L"  ProtocolsPerHandle failed for driver handle: 0x%lx\n", Status);
  }

  //
  // 2) Try LoadedImage on driver handle to get the image device path (if any)
  //
  EFI_LOADED_IMAGE_PROTOCOL *LoadedImage = NULL;
  Status = gBS->HandleProtocol (DriverBindingHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
  if (!EFI_ERROR (Status) && LoadedImage != NULL) {
    Print (L"  LoadedImage: ImageBase=%p FilePath=%p\n", LoadedImage->ImageBase, LoadedImage->FilePath);
    if (LoadedImage->FilePath != NULL) {
      CHAR16 *DpStr = MyDevicePathToStr (LoadedImage->FilePath);
      if (DpStr != NULL) {
        Print (L"    Driver image device path: %s\n", DpStr);
        FreePool (DpStr);
      } else {
        Print (L"    (Driver image device path -> MyDevicePathToStr returned NULL)\n");
      }
    } else {
      Print (L"    (LoadedImage has no FilePath)\n");
    }
  } else {
    Print (L"  No LoadedImageProtocol on driver handle (status=0x%lx)\n", Status);
  }

  //
  // 3) Try to explicitly bind this driver to the controller using ConnectController
  //
  Status = gBS->ConnectController (ControllerHandle, DriverBindingHandle, NULL, TRUE);
  Print (L"  ConnectController(controller=%p, driver=%p) returned 0x%lx\n", ControllerHandle, DriverBindingHandle, Status);

  Print (L"--- End dump/try ---\n");
}


//
// Helper: find first PCI handle matching Vendor/Device
//
STATIC
EFI_HANDLE
FindPciHandleByVendorDevice (
  IN UINT16 VendorId,
  IN UINT16 DeviceId,
  OUT EFI_PCI_IO_PROTOCOL **OutPciIo OPTIONAL
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN Count = 0;
  EFI_HANDLE Found = NULL;

  if (OutPciIo != NULL) {
    *OutPciIo = NULL;
  }

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &Count, &PciHandles);
  if (EFI_ERROR(Status) || Count == 0) {
    return NULL;
  }

  for (UINTN i = 0; i < Count; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol (PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) {
      continue;
    }

    UINT32 Vd = 0;
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x00, 1, &Vd);
    if (EFI_ERROR(Status)) {
      continue;
    }
    UINT16 v = (UINT16)(Vd & 0xffff);
    UINT16 d = (UINT16)((Vd >> 16) & 0xffff);
    if (v == VendorId && d == DeviceId) {
      Found = PciHandles[i];
      if (OutPciIo != NULL) {
        *OutPciIo = PciIo;
      }
      break;
    }
  }

  if (PciHandles) {
    FreePool (PciHandles);
  }
  return Found;
}

//
// Helper: scan DriverBinding handles and for the first that returns Supported()==EFI_SUCCESS
// call DumpDriverHandleAndTryBind(ControllerHandle, thatDriverHandle).
//
STATIC
VOID
FindDriverBindingThatSupportsControllerAndDump (
  IN EFI_HANDLE ControllerHandle
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *DriverHandles = NULL;
  UINTN DriverCount = 0;
  BOOLEAN FoundAny = FALSE;

  if (ControllerHandle == NULL) {
    Print (L"FindDriverBinding...: ControllerHandle is NULL\n");
    return;
  }

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiDriverBindingProtocolGuid, NULL, &DriverCount, &DriverHandles);
  if (EFI_ERROR(Status) || DriverCount == 0) {
    Print (L"No driver-binding handles found: 0x%lx\n", Status);
    return;
  }

  for (UINTN i = 0; i < DriverCount; ++i) {
    EFI_DRIVER_BINDING_PROTOCOL *DrvBinding = NULL;
    Status = gBS->HandleProtocol (DriverHandles[i], &gEfiDriverBindingProtocolGuid, (VOID**)&DrvBinding);
    if (EFI_ERROR(Status) || DrvBinding == NULL) {
      continue;
    }
	
	
	// previus version for first test 
    // Ask Supported() for this controller (NULL RemainingDevicePath is usual)
    Status = DrvBinding->Supported (DrvBinding, ControllerHandle, NULL);
    Print (L"  DriverBinding handle %p Supported() => 0x%lx\n", DriverHandles[i], Status);

    if (Status == EFI_SUCCESS) {
      // call the dump/try helper you already have
      Print (L"  -> Found supporting driver-binding handle %p; calling DumpDriverHandleAndTryBind()\n", DriverHandles[i]);
      DumpDriverHandleAndTryBind (ControllerHandle, DriverHandles[i]);
      FoundAny = TRUE;
      // optional: continue to find other drivers as well; for now break after first
      break;
    }
	
	
	/*
	Status = DrvBinding->Supported (DrvBinding, ControllerHandle, NULL);
    Print (L"  DriverBinding handle %p Supported() => 0x%lx\n", DriverHandles[i], Status);

	
	// new call method
	EFI_PCI_IO_PROTOCOL *TmpPciIo = NULL;
	EFI_HANDLE PciH = FindPciHandleByVendorDevice (0x106B, 0x003F, &TmpPciIo);
	if (PciH != NULL) {
	  // If you found the driver-binding handle earlier (e.g. 7E02ED98), call:
	  DumpDriverHandleAndTryBind (PciH, DriverBindingHandle);
	  // Or use FindDriverBindingThatSupportsControllerAndDump(PciH) to auto-discover
	}
	*/

	
  }

  if (!FoundAny) {
    Print (L"  No driver-binding reported Supported()==EFI_SUCCESS for this controller.\n");
  }

  if (DriverHandles) FreePool (DriverHandles);
}

//
// USAGE: where to call these from UefiMain()
// Example snippet to insert into UefiMain() at a convenient spot (after scanning PCI)
//

/*
  // Example: call from UefiMain after your PCI/HC scanning steps
  EFI_PCI_IO_PROTOCOL *TargetPciIo = NULL;
  EFI_HANDLE TargetPciHandle;

  // find handle for Vendor=0x106B Device=0x003F
  TargetPciHandle = FindPciHandleByVendorDevice (0x106B, 0x003F, &TargetPciIo);
  if (TargetPciHandle == NULL) {
    Print (L"Could not find PCI device 106B:003F\n");
  } else {
    Print (L"Found PCI handle %p for 106B:003F\n", TargetPciHandle);

    // 1) list all currently published UsbIo handles (shows which root-port/DevicePaths exist)
    ListAllUsbIoHandlesWithPortNodes ();

    // 2) automatically find driver-binding that claims support and dump it
    FindDriverBindingThatSupportsControllerAndDump (TargetPciHandle);

    // 3) Optionally re-diagnose the controller registers after trying to bind
    if (TargetPciIo != NULL) {
      // You already have DiagnosePortOnController, pass BarIndex and OpBase as you compute earlier.
      // Example: read CAPLENGTH / OpBase from BAR0 region - you probably have code for that.
      // DiagnosePortOnController (TargetPciIo, (UINT8)BarIndex, OpBase, 1);
    }
  }
*/


STATIC VOID TryUsbMassStorageInquiry (IN EFI_USB_IO_PROTOCOL *UsbIo);


// Main inspector: find PCI handle by vendor/device and enumerate attached USB devices
STATIC
VOID
InspectPciAndAttachedUsbDevices (
  IN UINT16 VendorId,
  IN UINT16 DeviceId
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN PciCount = 0;

  Print (L"--- Inspect PCI devices for Vendor=0x%04x Device=0x%04x ---\n", VendorId, DeviceId);

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Print (L"  No PCI handles found: 0x%lx\n", Status);
    return;
  }

  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol (PciHandles[i], &gEfiPciIoProtocolGuid, (VOID**)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) continue;

    // Read Vendor/Device dword
    UINT32 Vd = 0;
    Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x00, 1, &Vd);
    if (EFI_ERROR(Status)) continue;

    UINT16 v = (UINT16)(Vd & 0xffff);
    UINT16 d = (UINT16)((Vd >> 16) & 0xffff);
    if (v == VendorId && d == DeviceId) {
      // Found match
      Print (L"Found matching PCI handle: handle=%p (index=%u)\n", PciHandles[i], (UINT32)i);
      DumpPciBasicAndConfig (PciIo);

      //
      // Now scan EFI_USB_IO_PROTOCOL handles and match their DevicePath's PCI node
      //
      EFI_HANDLE *UsbHandles = NULL;
      UINTN UsbCount = 0;
      Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
      if (EFI_ERROR(Status)) {
        Print (L"  No UsbIo handles found: 0x%lx\n", Status);
      } else {
        Print (L"  Scanning %u EFI_USB_IO_PROTOCOL handles for device path PCI node matching this controller...\n", (UINT32)UsbCount);
        for (UINTN u = 0; u < UsbCount; ++u) {
          EFI_USB_IO_PROTOCOL *UsbIo = NULL;
          EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
          Status = gBS->HandleProtocol (UsbHandles[u], &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
          if (EFI_ERROR(Status) || UsbIo == NULL) continue;
		  
		  // existing device descriptor logging...
		  // I don't know if this works here before rest of the code
			TryUsbMassStorageInquiry (UsbIo);

          // get DevicePath
          Status = gBS->HandleProtocol (UsbHandles[u], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
          // If no DevicePath, fall back to printing protocols only
          if (DevPath == NULL) continue;

          // Parse devpath to find pci node device/function and compare with PCI location we already read
          // The PCI node in UEFI device path contains Device and Function bytes, but not Bus.
          // Get the Device/Function of the target PCI handle:
          UINTN Seg, Bus, Dev, Func;
          PciIo->GetLocation (PciIo, &Seg, &Bus, &Dev, &Func);
          // Compare by Device and Function bytes inside device path nodes
          if (DevicePathContainsPciNodeMatch (DevPath, (UINT8)Dev, (UINT8)Func)) {
            // match - print handle and device descriptors
            Print (L"  -> USB device handle %p appears to be attached under this PCI controller\n", UsbHandles[u]);
            CHAR16 *DpStr = MyDevicePathToStr (DevPath);
            if (DpStr) {
              Print (L"     DevicePath: %s\n", DpStr);
              FreePool (DpStr);
            }

            // Retrieve and print device descriptor and config/interface/endpoint info (reuse your code)
            {
              EFI_USB_DEVICE_DESCRIPTOR DevDesc;
              ZeroMem (&DevDesc, sizeof(DevDesc));
              Status = UsbIo->UsbGetDeviceDescriptor (UsbIo, &DevDesc);
              PrintStatus (L"    UsbGetDeviceDescriptor", Status);
              if (!EFI_ERROR(Status)) {
                Print (L"      bLength=%u bDescriptorType=%u bcdUSB=0x%04x idVendor=0x%04x idProduct=0x%04x bDeviceClass=0x%02x\n",
                       DevDesc.Length, DevDesc.DescriptorType, DevDesc.BcdUSB, DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
              }

              EFI_USB_CONFIG_DESCRIPTOR ConfigDesc;
              ZeroMem (&ConfigDesc, sizeof(ConfigDesc));
              Status = UsbIo->UsbGetConfigDescriptor (UsbIo, &ConfigDesc);
              PrintStatus (L"    UsbGetConfigDescriptor", Status);
              if (!EFI_ERROR(Status)) {
                Print (L"      wTotalLength=%u bNumInterfaces=%u bConfigurationValue=%u\n",
                       ConfigDesc.TotalLength, ConfigDesc.NumInterfaces, ConfigDesc.ConfigurationValue);
              }

              EFI_USB_INTERFACE_DESCRIPTOR IfDesc;
              ZeroMem (&IfDesc, sizeof(IfDesc));
              Status = UsbIo->UsbGetInterfaceDescriptor (UsbIo, &IfDesc);
              PrintStatus (L"    UsbGetInterfaceDescriptor", Status);
              if (!EFI_ERROR(Status)) {
                Print (L"      Interface: bInterfaceClass=0x%02x bInterfaceSubClass=0x%02x bInterfaceProtocol=0x%02x\n",
                       IfDesc.InterfaceClass, IfDesc.InterfaceSubClass, IfDesc.InterfaceProtocol);
              }

              // endpoints
              UINT8 EpIndex = 0;
              EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
              while (TRUE) {
                ZeroMem (&EpDesc, sizeof(EpDesc));
                Status = UsbIo->UsbGetEndpointDescriptor (UsbIo, EpIndex, &EpDesc);
                if (EFI_ERROR(Status)) {
                  if (EpIndex == 0) {
                    Print (L"      UsbGetEndpointDescriptor returned %r (no endpoints?)\n", Status);
                  } else {
                    Print (L"      End of endpoint list (index %u). Last status=%r\n", EpIndex, Status);
                  }
                  break;
                }
                Print (L"      Endpoint %u: addr=0x%02x attr=0x%02x maxp=%u interval=%u\n",
                       EpIndex, EpDesc.EndpointAddress, EpDesc.Attributes, EpDesc.MaxPacketSize, EpDesc.Interval);
                EpIndex++;
              }

              // If device class or interface class indicates hub (0x09), try to fetch hub descriptor
              if (DevDesc.DeviceClass == 0x09 || IfDesc.InterfaceClass == 0x09) {
                Print (L"    Device or interface indicates HUB class -> attempting to read hub descriptor\n");
                TryGetHubDescriptor (UsbIo);
              }
            } // device info block

          } // matched device path
        } // for each Usb handle

        FreePool (UsbHandles);
      } // had Usb handles

      // After printing the first matching PCI handle, continue to look for others if needed
    } // if vendor/device match
  } // for each pci handle

  if (PciHandles) FreePool (PciHandles);

  Print (L"--- End Inspect ---\n");
}

/*
  Perform a SCSI INQUIRY (read-only) using USB Mass Storage Bulk-Only Transport (BOT).
  Requires the target UsbIo handle (EFI_USB_IO_PROTOCOL*).
*/
STATIC
VOID
TryUsbMassStorageInquiry (
  IN EFI_USB_IO_PROTOCOL *UsbIo
  )
{
  EFI_STATUS Status;
  EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
  UINT8 epIdx = 0;
  UINT8 bulkOut = 0; // endpoint address for OUT (e.g. 0x01)
  UINT8 bulkIn  = 0; // endpoint address for IN  (e.g. 0x82)
  BOOLEAN foundOut = FALSE, foundIn = FALSE;

  if (UsbIo == NULL) {
    Print (L"TryUsbMassStorageInquiry: UsbIo is NULL\n");
    return;
  }

  // Find bulk IN and OUT endpoints by enumerating endpoint descriptors
  epIdx = 0;
  while (TRUE) {
    ZeroMem (&EpDesc, sizeof(EpDesc));
    Status = UsbIo->UsbGetEndpointDescriptor (UsbIo, epIdx, &EpDesc);
    if (EFI_ERROR(Status)) {
      break;
    }
    // bmAttributes low 2 bits describe transfer type (0=control,1=iso,2=bulk,3=interrupt)
    UINT8 type = EpDesc.Attributes & 0x03;
    UINT8 addr = EpDesc.EndpointAddress & 0xFF;
    if (type == USB_ENDPOINT_TYPE_BULK) {
      if (addr & 0x80) {
        bulkIn = addr;
        foundIn = TRUE;
      } else {
        bulkOut = addr;
        foundOut = TRUE;
      }
    }
    epIdx++;
  }

  if (!foundIn || !foundOut) {
    Print (L"  Bulk endpoints not found (IN=%d OUT=%d). Cannot perform BOT. IN=0x%02x OUT=0x%02x\n",
           foundIn, foundOut, bulkIn, bulkOut);
    return;
  }

  Print (L"  Found bulk endpoints: OUT=0x%02x IN=0x%02x\n", bulkOut, bulkIn);

  //
  // Build CBW (31 bytes)
  //
  // CBW structure (packed):
  // dCBWSignature (0x43425355), dCBWTag, dCBWDataTransferLength, bmCBWFlags, bCBWLUN, bCBWCBLength, CBWCB[16]
  //
  #pragma pack(push,1)
  typedef struct {
    UINT32  dCBWSignature;
    UINT32  dCBWTag;
    UINT32  dCBWDataTransferLength;
    UINT8   bmCBWFlags;
    UINT8   bCBWLUN;
    UINT8   bCBWCBLength;
    UINT8   CBWCB[16];
  } CBW;
  #pragma pack(pop)

  #pragma pack(push,1)
  typedef struct {
    UINT32 dCSWSignature;
    UINT32 dCSWTag;
    UINT32 dCSWDataResidue;
    UINT8  bCSWStatus;
  } CSW;
  #pragma pack(pop)

  CBW cbw;
  CSW csw;
  UINTN TransLen;
  UINT8 InquiryBuf[36];
  UINT32 tag = 0x1A5A1234; // arbitrary tag

  // Prepare SCSI INQUIRY CDB (6 bytes)
  UINT8 InquiryCdb[6];
  ZeroMem (InquiryCdb, sizeof(InquiryCdb));
  InquiryCdb[0] = 0x12; // INQUIRY
  InquiryCdb[1] = 0x00; // EVPD=0, LUN in high bits if needed
  InquiryCdb[2] = 0x00; // Page code
  InquiryCdb[3] = 0x00;
  InquiryCdb[4] = sizeof(InquiryBuf); // Allocation length (36)
  InquiryCdb[5] = 0x00;

  ZeroMem (&cbw, sizeof(cbw));
  cbw.dCBWSignature = 0x43425355; // 'USBC' little-endian
  cbw.dCBWTag = tag;
  cbw.dCBWDataTransferLength = sizeof(InquiryBuf);
  cbw.bmCBWFlags = 0x80; // bit7 = 1 => data-in (device -> host)
  cbw.bCBWLUN = 0; // LUN 0
  cbw.bCBWCBLength = 6; // INQUIRY CDB length
  CopyMem (cbw.CBWCB, InquiryCdb, 6);

  // 1) Send CBW (bulk OUT)
  TransLen = sizeof(cbw);
  UINT32 ret_status = 0;
  Status = UsbIo->UsbBulkTransfer (UsbIo, bulkOut, &cbw, &TransLen, TIMEOUT_MS, &ret_status);
  PrintStatus (L"  UsbBulkTransfer(CBW OUT)", Status);
  if (EFI_ERROR(Status)) {
    Print (L"    Failed to send CBW\n");
    return;
  }
  if (TransLen != sizeof(cbw)) {
    Print (L"    Warning: CBW bytes sent mismatch: sent=%u expected=%u\n", (UINT32)TransLen, (UINT32)sizeof(cbw));
  }

  // 2) Read DATA-IN (INQUIRY response) (bulk IN)
  ZeroMem (InquiryBuf, sizeof(InquiryBuf));
  TransLen = sizeof(InquiryBuf);
  Status = UsbIo->UsbBulkTransfer (UsbIo, bulkIn, InquiryBuf, &TransLen, TIMEOUT_MS, &ret_status);
  PrintStatus (L"  UsbBulkTransfer(INQUIRY DATA-IN)", Status);
  if (!EFI_ERROR(Status)) {
    Print (L"    INQUIRY returned %u bytes\n", (UINT32)TransLen);
    DumpHex (L"    INQUIRY data", InquiryBuf, (TransLen > 64) ? 64 : TransLen);
    // Print some readable fields: Vendor/Prod/Revision
    if (TransLen >= 36) {
      CHAR8 vend[9]; CHAR8 prod[17]; CHAR8 rev[5];
      ZeroMem (vend, sizeof(vend)); ZeroMem (prod, sizeof(prod)); ZeroMem (rev, sizeof(rev));
      CopyMem (vend, &InquiryBuf[8], 8);
      CopyMem (prod, &InquiryBuf[16], 16);
      CopyMem (rev, &InquiryBuf[32], 4);
      Print (L"    INQUIRY: Vendor=\"%a\" Product=\"%a\" Rev=\"%a\"\n", vend, prod, rev);
    }
  } else {
    Print (L"    DATA-IN stage failed (Status=0x%lx). Attempting to read CSW anyway.\n", Status);
  }

  // 3) Read CSW (bulk IN, 13 bytes)
  ZeroMem (&csw, sizeof(csw));
  TransLen = sizeof(csw);
  Status = UsbIo->UsbBulkTransfer (UsbIo, bulkIn, &csw, &TransLen, TIMEOUT_MS, &ret_status);
  PrintStatus (L"  UsbBulkTransfer(CSW IN)", Status);
  if (EFI_ERROR(Status)) {
    Print (L"    Failed to read CSW: 0x%lx\n", Status);
    return;
  }
  if (TransLen < sizeof(CSW)) {
    Print (L"    CSW too short (%u bytes)\n", (UINT32)TransLen);
  } else {
    // validate signature and tag
    if (csw.dCSWSignature != 0x53425355) { // 'USBS'
      Print (L"    CSW signature mismatch: 0x%08x\n", csw.dCSWSignature);
    } else if (csw.dCSWTag != tag) {
      Print (L"    CSW tag mismatch: got 0x%08x expected 0x%08x\n", csw.dCSWTag, tag);
    } else {
      Print (L"    CSW status = 0x%02x DataResidue=0x%08x\n", csw.bCSWStatus, csw.dCSWDataResidue);
      if (csw.bCSWStatus == 0) {
        Print (L"    SCSI INQUIRY completed successfully (no writes performed)\n");
      } else {
        Print (L"    SCSI INQUIRY failed (bCSWStatus=0x%02x)\n", csw.bCSWStatus);
      }
    }
  }
}


// Health-check helpers: call after your main sequence to re-check ports on a controller.
// Requires: FindPciHandleByVendorDevice(), MyDevicePathToStr(), TryUsbMassStorageInquiry(),
//           DevicePathContainsPciNodeMatch() or equivalent device-path parsing helpers.
// If you already have those, the code will reuse them; otherwise this file contains its own helpers.

STATIC
BOOLEAN
DevicePathHasPciAndUsbPort (
  IN CONST EFI_DEVICE_PATH_PROTOCOL *DevicePath,
  IN UINT8                          TargetDev,
  IN UINT8                          TargetFunc,
  IN UINT8                          TargetParentPort
  )
{
  CONST EFI_DEVICE_PATH_PROTOCOL *Node = DevicePath;
  UINT8 foundPciDev = 0;
  UINT8 foundPciFunc = 0;
  UINT8 foundUsbParentPort = 0;

  if (DevicePath == NULL) {
    return FALSE;
  }

  while (Node != NULL) {
    UINT8 Type = Node->Type;
    UINT8 SubType = Node->SubType;
    UINT16 NodeLen = (UINT16)(Node->Length[0] | (Node->Length[1] << 8));
    if (Type == DP_TYPE_END && SubType == DP_SUBTYPE_END_ENTIRE) break;
    if (NodeLen < 4) break;

    if (Type == DP_TYPE_HARDWARE && SubType == DP_SUBTYPE_PCI && NodeLen >= 6) {
      UINT8 *raw = (UINT8*)Node;
      UINT8 Dev = raw[4];
      UINT8 Func = raw[5];
      if (Dev == TargetDev && Func == TargetFunc) {
        foundPciDev = 1;
        foundPciFunc = 1;
      }
    }

    if (Type == DP_TYPE_MESSAGING && SubType == DP_SUBTYPE_USB && NodeLen >= 6) {
      UINT8 *raw = (UINT8*)Node;
      UINT8 ParentPort = raw[4];
      // Interface number is raw[5] but we only care about ParentPort
      if (ParentPort == TargetParentPort) {
        foundUsbParentPort = 1;
      }
    }

    Node = (CONST EFI_DEVICE_PATH_PROTOCOL *) ((CONST UINT8 *) Node + NodeLen);
  }

  // We want both a pci node and a USB(port,iface) node that match
  return (foundPciDev && foundUsbParentPort);
}

STATIC
EFI_HANDLE
FindUsbIoHandleForPciPort (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               ParentPortNumber
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbCount = 0;
  EFI_HANDLE Found = NULL;
  UINTN Seg, Bus, Dev, Func;

  if (PciIo == NULL) {
    return NULL;
  }

  PciIo->GetLocation (PciIo, &Seg, &Bus, &Dev, &Func);

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
  if (EFI_ERROR (Status) || UsbCount == 0) {
    if (UsbHandles) FreePool (UsbHandles);
    return NULL;
  }

  for (UINTN i = 0; i < UsbCount; ++i) {
    EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
    Status = gBS->HandleProtocol (UsbHandles[i], &gEfiDevicePathProtocolGuid, (VOID**)&DevPath);
    if (EFI_ERROR (Status) || DevPath == NULL) {
      continue;
    }

    if (DevicePathHasPciAndUsbPort (DevPath, (UINT8)Dev, (UINT8)Func, ParentPortNumber)) {
      Found = UsbHandles[i];
      break;
    }
  }

  if (UsbHandles) FreePool (UsbHandles);
  return Found;
}

/**
  Check health/status of ports on the controller identified by Vendor/Device.
  Prints details and attempts safe device-level reads for any published UsbIo children on each port.
*/
STATIC
VOID
CheckControllerPortHealthByVendorDevice (
  IN UINT16 VendorId,
  IN UINT16 DeviceId
  )
{
  EFI_STATUS Status;
  EFI_PCI_IO_PROTOCOL *PciIo = NULL;
  EFI_HANDLE PciHandle = FindPciHandleByVendorDevice (VendorId, DeviceId, &PciIo);
  if (PciHandle == NULL || PciIo == NULL) {
    Print (L"CheckControllerPortHealth: PCI device %04x:%04x not found\n", VendorId, DeviceId);
    return;
  }

  Print (L"\n--- Controller health check for PCI %04x:%04x (handle=%p) ---\n", VendorId, DeviceId, PciHandle);

  // Probe BARs for capability region / OpBase
  BOOLEAN FoundMmio = FALSE;
  UINT8 BarIndex;
  for (BarIndex = 0; BarIndex < MAX_BARS; ++BarIndex) {
    UINT8 CapLen = 0xFF;
    Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLen);
    if (EFI_ERROR (Status)) {
      continue;
    }
    if (CapLen == 0xFF || CapLen == 0x00) {
      continue;
    }
    FoundMmio = TRUE;
    UINTN OpBase = (UINTN)CapLen;

    // Read main registers
    UINT32 UsbCmd = 0, UsbSts = 0, FrIndex = 0, AsyncList = 0, PeriodicBase = 0, HcSParams = 0;
    PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x00, 1, &UsbCmd);
    PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x04, 1, &UsbSts);
    PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x0C, 1, &FrIndex);
    PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x18, 1, &AsyncList);
    PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, OpBase + 0x10, 1, &PeriodicBase);
    PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, 0x04, 1, &HcSParams);

    Print (L"  BAR%u OpBase=0x%03x  USBCMD=0x%08x USBSTS=0x%08x FRINDEX=0x%08x\n", (UINT32)BarIndex, (UINT32)OpBase, UsbCmd, UsbSts, FrIndex);
    Print (L"    ASYNCLISTADDR=0x%08x PERIODICLISTBASE=0x%08x HCSPARAMS=0x%08x\n", AsyncList, PeriodicBase, HcSParams);

    // Determine port count
    UINT32 NPorts = (HcSParams & 0x0000000F);
    if (NPorts == 0) {
      NPorts = MAX_PORTS_SAFE;
    }
    if (NPorts > MAX_PORTS_SAFE) NPorts = MAX_PORTS_SAFE;

    Print (L"    Probing up to %u ports (HCSPARAMS lower nibble => %u)\n", (UINT32)NPorts, (UINT32)(HcSParams & 0xF));

    for (UINT32 PortIndex = 1; PortIndex <= NPorts; ++PortIndex) {
      UINT32 PortSc = 0;
      UINTN PortOffset = OpBase + 0x44 + (PortIndex - 1) * 4;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
      if (EFI_ERROR (Status)) {
        // stop if we can't read more ports
        break;
      }

      UINTN CCS = (PortSc >> 0) & 1;
      UINTN CSC = (PortSc >> 1) & 1;
      UINTN PE  = (PortSc >> 2) & 1;
      UINTN PEchg = (PortSc >> 3) & 1;
      UINTN PO  = (PortSc >> 13) & 1; // approximate position
      UINTN Speed = (PortSc >> 26) & 3;

      Print (L"    PORT%u: PORTSC=0x%08x (CCS=%u CSC=%u PE=%u PEchg=%u PO=%u Speed=%u)\n",
             (UINT32)PortIndex, PortSc, CCS, CSC, PE, PEchg, PO, Speed);

      // Find published UsbIo handle corresponding to this PCI port (if published)
      EFI_HANDLE UsbHandle = FindUsbIoHandleForPciPort (PciIo, (UINT8)PortIndex);
      if (UsbHandle != NULL) {
        Print (L"      -> Found published EFI_USB_IO handle %p for Port%u\n", UsbHandle, (UINT32)PortIndex);

        // print the device path for the Usb handle
        EFI_DEVICE_PATH_PROTOCOL *Dp = NULL;
        Status = gBS->HandleProtocol (UsbHandle, &gEfiDevicePathProtocolGuid, (VOID**)&Dp);
        if (!EFI_ERROR (Status) && Dp != NULL) {
          CHAR16 *DpStr = MyDevicePathToStr (Dp);
          if (DpStr != NULL) {
            Print (L"         DevicePath: %s\n", DpStr);
            FreePool (DpStr);
          }
        }

        // Attempt safe device-level probe: device descriptor
        EFI_USB_IO_PROTOCOL *UsbIo = NULL;
        Status = gBS->HandleProtocol (UsbHandle, &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
        if (!EFI_ERROR (Status) && UsbIo != NULL) {
          EFI_USB_DEVICE_DESCRIPTOR DevDesc;
          ZeroMem (&DevDesc, sizeof(DevDesc));
          Status = UsbIo->UsbGetDeviceDescriptor (UsbIo, &DevDesc);
          PrintStatus (L"         UsbGetDeviceDescriptor", Status);
          if (!EFI_ERROR (Status)) {
            Print (L"           idVendor=0x%04x idProduct=0x%04x bDeviceClass=0x%02x\n",
                   DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass);
            // If mass-storage, perform read-only inquiry test (non-destructive)
            if (DevDesc.DeviceClass == 0x08) {
              Print (L"           Device class indicates Mass Storage -> running read-only INQUIRY test\n");
              TryUsbMassStorageInquiry (UsbIo);
            }
          }
        } else {
          Print (L"         Failed to open EFI_USB_IO on handle %p: 0x%lx\n", UsbHandle, Status);
        }
      } else {
        // No published UsbIo handle for this port
        if (CCS && !PE && !PO) {
          Print (L"      -> Port%u CONNECTED but NOT ENABLED and EHCI owns it (un-enumerated)\n", (UINT32)PortIndex);
        } else if (CCS && !PE && PO) {
          Print (L"      -> Port%u CONNECTED but PortOwner=1 (companion controller owns it)\n", (UINT32)PortIndex);
        } else if (!CCS) {
          Print (L"      -> Port%u no device connected\n", (UINT32)PortIndex);
        } else {
          Print (L"      -> Port%u connected but no published UsbIo handle (state PE=%u PO=%u)\n", (UINT32)PortIndex, (UINT32)PE, (UINT32)PO);
        }
      }
    } // per port

    // Completed this BAR/reg region - if you only want first match, break; we continue to next BAR by design
    // If you prefer to only check first CAP region, uncomment next line:
    // break;
  } // per BAR

  if (!FoundMmio) {
    Print (L"  No accessible MMIO BAR/cap region found for controller (skipped).\n");
  }

  Print (L"--- End controller health check ---\n\n");
}

// Small helper: attempt a tiny IN transfer to an endpoint (non-destructive).
// If endpoint is IN and of bulk/interrupt type, try to read up to BufSize bytes.
// Returns EFI_SUCCESS on read success (even if zero bytes).
STATIC
EFI_STATUS
TrySmallInTransfer (
  IN EFI_USB_IO_PROTOCOL *UsbIo,
  IN UINT8               EndpointAddr,
  IN UINTN               BufSize
  )
{
	
	UINT32 ret_status = 0;
	
  if (UsbIo == NULL) return EFI_INVALID_PARAMETER;
  UINT8 *Buf = AllocatePool (BufSize);
  if (Buf == NULL) return EFI_OUT_OF_RESOURCES;
  ZeroMem (Buf, BufSize);
  UINTN Transfer = BufSize;
  EFI_STATUS Status = UsbIo->UsbBulkTransfer (UsbIo, EndpointAddr, Buf, &Transfer, TIMEOUT_MS, &ret_status);
  if (EFI_ERROR(Status)) {
    // If it's an interrupt endpoint, try UsbIo->UsbInterruptTransfer instead (some stacks use this)
    // But do not spam; only try once more via Interrupt API if Bulk failed and endpoint is interrupt.
    // We will attempt an interrupt transfer if bulk failed and the endpoint addr indicates IN.
    // Caller must decide which API to call; here we only try bulk.
    FreePool (Buf);
    return Status;
  }
  Print (L"    Small IN read returned %u bytes\n", (UINT32)Transfer);
  if (Transfer > 0) {
    DumpHex (L"    Payload (first 64 bytes)", Buf, (Transfer > 64) ? 64 : Transfer);
  }
  FreePool (Buf);
  return EFI_SUCCESS;
}

// Probe one published UsbIo handle (safe, read-only checks).
STATIC
VOID
DoDeviceLevelProbe (
  IN EFI_USB_IO_PROTOCOL *UsbIo
  )
{
  EFI_STATUS Status;
  UINT32 ret_status = 0;


  if (UsbIo == NULL) {
    Print (L"DoDeviceLevelProbe: UsbIo == NULL\n");
    return;
  }

  // 1) Device descriptor
  EFI_USB_DEVICE_DESCRIPTOR DevDesc;
  ZeroMem (&DevDesc, sizeof(DevDesc));
  Status = UsbIo->UsbGetDeviceDescriptor (UsbIo, &DevDesc);
  PrintStatus (L"  UsbGetDeviceDescriptor", Status);
  if (!EFI_ERROR (Status)) {
    Print (L"    idVendor=0x%04x idProduct=0x%04x bDeviceClass=0x%02x bcdUSB=0x%04x\n",
           DevDesc.IdVendor, DevDesc.IdProduct, DevDesc.DeviceClass, DevDesc.BcdUSB);
  } else {
    // can't proceed much if we can't read device descriptor
    return;
  }

  // 2) Config descriptor
  EFI_USB_CONFIG_DESCRIPTOR ConfigDesc;
  ZeroMem (&ConfigDesc, sizeof(ConfigDesc));
  Status = UsbIo->UsbGetConfigDescriptor (UsbIo, &ConfigDesc);
  PrintStatus (L"  UsbGetConfigDescriptor", Status);
  if (!EFI_ERROR (Status)) {
    Print (L"    wTotalLength=%u bNumInterfaces=%u\n", ConfigDesc.TotalLength, ConfigDesc.NumInterfaces);
  }

  // 3) Interface descriptor
  EFI_USB_INTERFACE_DESCRIPTOR IfDesc;
  ZeroMem (&IfDesc, sizeof(IfDesc));
  Status = UsbIo->UsbGetInterfaceDescriptor (UsbIo, &IfDesc);
  PrintStatus (L"  UsbGetInterfaceDescriptor", Status);
  if (!EFI_ERROR (Status)) {
    Print (L"    Interface: Class=0x%02x SubClass=0x%02x Protocol=0x%02x\n",
           IfDesc.InterfaceClass, IfDesc.InterfaceSubClass, IfDesc.InterfaceProtocol);
  }

  // 4) Enumerate endpoints and remember bulk/interrupt IN endpoints
  UINT8 EpIndex = 0;
  UINT8 BulkInEp = 0;
  UINT8 BulkOutEp = 0;
  UINT8 IntInEp = 0;
  EFI_USB_ENDPOINT_DESCRIPTOR EpDesc;
  while (TRUE) {
    ZeroMem (&EpDesc, sizeof(EpDesc));
    Status = UsbIo->UsbGetEndpointDescriptor (UsbIo, EpIndex, &EpDesc);
    if (EFI_ERROR (Status)) {
      break;
    }
    UINT8 attrType = EpDesc.Attributes & 0x03;
    Print (L"    Endpoint %u: addr=0x%02x attr=0x%02x maxp=%u interval=%u\n",
           EpIndex, EpDesc.EndpointAddress, EpDesc.Attributes, EpDesc.MaxPacketSize, EpDesc.Interval);

    if (attrType == USB_ENDPOINT_TYPE_BULK) {
      if (EpDesc.EndpointAddress & 0x80) BulkInEp = EpDesc.EndpointAddress;
      else BulkOutEp = EpDesc.EndpointAddress;
    } else if (attrType == USB_ENDPOINT_TYPE_INTERRUPT) {
      if (EpDesc.EndpointAddress & 0x80) IntInEp = EpDesc.EndpointAddress;
    }
    EpIndex++;
  }

  // 5) Safe control GET_STATUS (device)
  {
    EFI_USB_DEVICE_REQUEST DevReq;
    UINT8 StatusBuf[2];
    UINT32 UsbSt = 0;
    ZeroMem (&DevReq, sizeof(DevReq));
    DevReq.RequestType = USB_ENDPOINT_DIR_IN | USB_REQ_TYPE_STANDARD | USB_TARGET_DEVICE;
    DevReq.Request = USB_REQ_GET_STATUS;
    DevReq.Value = 0;
    DevReq.Index = 0;
    DevReq.Length = 2;
    ZeroMem (StatusBuf, sizeof(StatusBuf));
    Status = UsbIo->UsbControlTransfer (UsbIo, &DevReq, EfiUsbDataIn, TIMEOUT_MS, StatusBuf, sizeof(StatusBuf), &UsbSt);
    PrintStatus (L"  UsbControlTransfer(GET_STATUS)", Status);
    if (!EFI_ERROR (Status)) {
      Print (L"    GET_STATUS data: %02x %02x\n", StatusBuf[0], StatusBuf[1]);
    }
  }

  // 6) If Mass Storage class -> use BOT INQUIRY
  if (DevDesc.DeviceClass == 0x08 || IfDesc.InterfaceClass == 0x08) {
    Print (L"  Device is mass storage -> running BOT/SCSI INQUIRY test\n");
    TryUsbMassStorageInquiry (UsbIo);
    return;
  }

  // 7) Otherwise, if a bulk-IN endpoint exists try a tiny read to test data path
  if (BulkInEp) {
    Print (L"  Trying small bulk-IN test on 0x%02x\n", BulkInEp);
    Status = TrySmallInTransfer (UsbIo, BulkInEp, 64);
    PrintStatus (L"  TrySmallInTransfer(bulk-in)", Status);
  } else if (IntInEp) {
    // Try interrupt transfer (use UsbInterruptTransfer) if supported by your USB_IO implementation.
    // We'll attempt a small interrupt read via UsbBulkTransfer first (some stacks accept it), else try interrupt API.
    Print (L"  Trying small interrupt-IN test on 0x%02x\n", IntInEp);
    UINT8 *Buf = AllocatePool (64);
    if (Buf != NULL) {
      ZeroMem (Buf, 64);
      UINTN Len = 64;
      //Status = UsbIo->UsbInterruptTransfer (UsbIo, IntInEp, Buf, &Len, TIMEOUT_MS);
      //PrintStatus (L"  UsbInterruptTransfer(INT-IN)", Status);
      //if (!EFI_ERROR (Status) && Len > 0) {
      //  DumpHex (L"    INT-IN payload", Buf, (Len > 64) ? 64 : Len);
      //}
	  
	  // Try synchronous interrupt transfer if available; otherwise try bulk-in as fallback.
		if (UsbIo->UsbSyncInterruptTransfer != NULL) {
			//UINTN Len = 64;
			Status = UsbIo->UsbSyncInterruptTransfer (UsbIo, IntInEp, Buf, &Len, TIMEOUT_MS, &ret_status);
			PrintStatus (L"  UsbSyncInterruptTransfer(INT-IN)", Status);
			Print(L" Status %u \n", ret_status);
			if (!EFI_ERROR(Status) && Len > 0) {
				DumpHex (L"    INT-IN payload", Buf, (Len > 64) ? 64 : Len);
			} else if (EFI_ERROR(Status)) {
				Print (L"    UsbSyncInterruptTransfer failed: 0x%lx\n", Status);
				// optional fallback to bulk
				UINTN BulkLen = 64;
				EFI_STATUS BulkStatus = UsbIo->UsbBulkTransfer (UsbIo, IntInEp, Buf, &BulkLen, TIMEOUT_MS, &ret_status);
				PrintStatus (L"    Fallback UsbBulkTransfer(INT-IN)", BulkStatus);
				Print(L" Status %u \n", ret_status);
				if (!EFI_ERROR(BulkStatus) && BulkLen > 0) {
					DumpHex (L"    Fallback payload", Buf, (BulkLen > 64) ? 64 : BulkLen);
				}
			}
		} else if (UsbIo->UsbAsyncInterruptTransfer != NULL) {
			// Sync API not present — try async as a best-effort diagnostic.
			// We will perform a single async transfer with a very simple completion callback stub.
			typedef struct { EFI_EVENT Event; EFI_STATUS Status; UINTN Len; } ASYNC_CTX;
			ASYNC_CTX ctx;
			ctx.Event = NULL;
			ctx.Status = EFI_NOT_READY;
			ctx.Len = 0;

			// Create an event for the completion callback
			Status = gBS->CreateEvent (EVT_NOTIFY_SIGNAL, TPL_CALLBACK, NULL, NULL, &ctx.Event);
			if (EFI_ERROR(Status)) {
				Print (L"    Failed to create event for async interrupt: 0x%lx\n", Status);
			} else {
				// Define a tiny callback that signals event (we can't pass context pointer in prototype easily here)
				// NOTE: proper async requires a function pointer with context; if your environment doesn't support
				// a simple callback capture, skip async attempt.
				// For brevity we will not implement a full async callback here; instead report that async is available.
				Print (L"    UsbSyncInterruptTransfer not present; UsbAsyncInterruptTransfer is available but async\n");
				Print (L"    diagnostic using async transfers is not implemented in this helper.\n");
				gBS->CloseEvent (ctx.Event);
			}
		} else {
			// no interrupt APIs available; try bulk-in as last resort
			UINTN BulkLen = 64;
			Status = UsbIo->UsbBulkTransfer (UsbIo, IntInEp, Buf, &BulkLen, TIMEOUT_MS, &ret_status);
			PrintStatus (L"  UsbBulkTransfer(INT-IN) fallback", Status);
			Print(L" Status %u \n", ret_status);
			if (!EFI_ERROR(Status) && BulkLen > 0) {
				DumpHex (L"    Bulk-IN payload", Buf, (BulkLen > 64) ? 64 : BulkLen);
			}
		}

	  
      FreePool (Buf);
    }
  } else {
    Print (L"  No IN endpoints to test data path (bulk-in or int-in not found)\n");
  }
}

// Run probes for a set of port numbers on a controller by vendor/device.
// Ports are passed as variable-length argument list terminated by 0 (or pass count and array).
STATIC
VOID
ProbeControllerPortsSimple (
  IN UINT16 VendorId,
  IN UINT16 DeviceId,
  IN UINT8  Port1,
  IN UINT8  Port2
  )
{
  EFI_STATUS Status;
  EFI_PCI_IO_PROTOCOL *PciIo = NULL;
  EFI_HANDLE PciH = FindPciHandleByVendorDevice (VendorId, DeviceId, &PciIo);
  if (PciH == NULL || PciIo == NULL) {
    Print (L"ProbeControllerPortsSimple: controller %04x:%04x not found\n", VendorId, DeviceId);
    return;
  }

  Print (L"\n--- Probing controller ports: %04x:%04x ports %u,%u ---\n", VendorId, DeviceId, Port1, Port2);

  // Re-run per-port health read similar to CheckControllerPortHealthByVendorDevice but only for specified ports.
  // Find MMIO opbase like before:
  BOOLEAN Found = FALSE;
  for (UINT8 BarIndex = 0; BarIndex < MAX_BARS && !Found; BarIndex++) {
    UINT8 CapLen = 0xFF;
    Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLen);
    if (EFI_ERROR(Status)) continue;
    if (CapLen == 0xFF || CapLen == 0x00) continue;
    UINTN OpBase = (UINTN)CapLen;
    Found = TRUE;

    // For each requested port, read PORTSC and try to find published UsbIo handle
    UINT8 ports[2] = { Port1, Port2 };
    for (UINTN i = 0; i < 2; ++i) {
      UINT8 p = ports[i];
      if (p == 0) continue;
      UINT32 PortSc = 0;
      UINTN PortOffset = OpBase + 0x44 + (p - 1) * 4;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortSc);
      if (EFI_ERROR(Status)) {
        Print (L"  Failed to read PORT%u: 0x%lx\n", (UINT32)p, Status);
        continue;
      }
      Print (L"  PORT%u PORTSC=0x%08x (CCS=%u PE=%u PO=%u)\n", (UINT32)p, PortSc, (UINT32)(PortSc & 1), (UINT32)((PortSc>>2)&1), (UINT32)((PortSc>>13)&1));

      // Try to find published UsbIo for this pci port
      EFI_HANDLE UsbH = FindUsbIoHandleForPciPort (PciIo, p);
      if (UsbH != NULL) {
        Print (L"    Published UsbIo handle %p for Port%u\n", UsbH, (UINT32)p);
        EFI_USB_IO_PROTOCOL *UsbIo = NULL;
        Status = gBS->HandleProtocol (UsbH, &gEfiUsbIoProtocolGuid, (VOID**)&UsbIo);
        if (EFI_ERROR(Status) || UsbIo == NULL) {
          Print (L"    Failed to open UsbIo protocol: 0x%lx\n", Status);
        } else {
          // run the device probe
          DoDeviceLevelProbe (UsbIo);
        }
      } else {
        Print (L"    No published UsbIo handle for Port%u\n", (UINT32)p);
      }
    }
  } // bar loop

  if (!Found) {
    Print (L"  No MMIO capability region found for controller to probe ports.\n");
  }

  Print (L"--- End probe ---\n\n");
}


#include <Protocol/DriverBinding.h>
#include <Protocol/LoadedImage.h>

/** Print all protocol GUID pointers installed on a handle (controller). */
STATIC
VOID
DumpProtocolsOnHandle (
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;
  EFI_GUID **Protocols = NULL;
  UINTN Count = 0;

  if (Handle == NULL) {
    Print (L"DumpProtocolsOnHandle: handle is NULL\n");
    return;
  }

  Status = gBS->ProtocolsPerHandle (Handle, &Protocols, &Count);
  if (EFI_ERROR(Status) || Protocols == NULL) {
    Print (L"  ProtocolsPerHandle failed for handle %p: 0x%lx\n", Handle, Status);
    return;
  }

  Print (L"  Handle %p has %u protocols installed:\n", Handle, (UINT32)Count);
  for (UINTN i = 0; i < Count; ++i) {
    if (Protocols[i] == NULL) {
      Print (L"    [%u] NULL\n", (UINT32)i);
    } else if (CompareGuid (Protocols[i], &gEfiPciIoProtocolGuid)) {
      Print (L"    [%u] EFI_PCI_IO_PROTOCOL\n", (UINT32)i);
    } else if (CompareGuid (Protocols[i], &gEfiUsb2HcProtocolGuid)) {
      Print (L"    [%u] EFI_USB2_HC_PROTOCOL\n", (UINT32)i);
    } else if (CompareGuid (Protocols[i], &gEfiUsbHcProtocolGuid)) {
      Print (L"    [%u] EFI_USB_HC_PROTOCOL\n", (UINT32)i);
    } else if (CompareGuid (Protocols[i], &gEfiUsbIoProtocolGuid)) {
      Print (L"    [%u] EFI_USB_IO_PROTOCOL\n", (UINT32)i);
    } else {
      // Fall back to printing pointer value; converting GUID->string is verbose/fragile here.
      Print (L"    [%u] GUID ptr=%p\n", (UINT32)i, (VOID*)Protocols[i]);
    }
  }

  FreePool (Protocols);
}

/** List all loaded images and their device path (so you can map DXE modules). */
STATIC
VOID
ListLoadedImages (VOID)
{
  EFI_STATUS Status;
  EFI_HANDLE *Handles = NULL;
  UINTN Count = 0;

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiLoadedImageProtocolGuid, NULL, &Count, &Handles);
  if (EFI_ERROR(Status) || Count == 0) {
    Print (L"ListLoadedImages: no LoadedImage handles found (status=0x%lx)\n", Status);
    if (Handles) FreePool (Handles);
    return;
  }

  Print (L"--- LoadedImage handles (%u) ---\n", (UINT32)Count);
  for (UINTN i = 0; i < Count; ++i) {
    EFI_LOADED_IMAGE_PROTOCOL *Li = NULL;
    Status = gBS->HandleProtocol (Handles[i], &gEfiLoadedImageProtocolGuid, (VOID**)&Li);
    if (EFI_ERROR(Status) || Li == NULL) {
      Print (L" [%u] Handle=%p LoadedImage protocol open failed: 0x%lx\n", (UINT32)i, Handles[i], Status);
      continue;
    }
    Print (L" [%u] Handle=%p ImageBase=%p\n", (UINT32)i, Handles[i], Li->ImageBase);
    if (Li->FilePath != NULL) {
      CHAR16 *s = MyDevicePathToStr (Li->FilePath);
      if (s != NULL) {
        Print (L"     FilePath: %s\n", s);
        FreePool (s);
      } else {
        Print (L"     FilePath: (MyDevicePathToStr returned NULL)\n");
      }
    } else {
      Print (L"     FilePath: (NULL)\n");
    }
  }
  FreePool (Handles);
  Print (L"--- end LoadedImage list ---\n");
}

/**
  Try to Start() every driver-binding handle that reports Supported()==EFI_SUCCESS
  for the provided ControllerHandle. Print the Start() return value and the driver's
  LoadedImage path (so you can find the DXE in your firmware tree).
*/
STATIC
VOID
TryStartAllSupportingDrivers (
  IN EFI_HANDLE ControllerHandle
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *DriverBindings = NULL;
  UINTN Count = 0;

  if (ControllerHandle == NULL) {
    Print (L"TryStartAllSupportingDrivers: controller handle NULL\n");
    return;
  }

  Status = gBS->LocateHandleBuffer (ByProtocol, &gEfiDriverBindingProtocolGuid, NULL, &Count, &DriverBindings);
  if (EFI_ERROR(Status) || Count == 0) {
    Print (L"  No driver-binding handles found: 0x%lx\n", Status);
    if (DriverBindings) FreePool (DriverBindings);
    return;
  }

  Print (L"--- Trying Start() on drivers that Supported() this controller ---\n");
  for (UINTN i = 0; i < Count; ++i) {
    EFI_DRIVER_BINDING_PROTOCOL *Drv = NULL;
    Status = gBS->HandleProtocol (DriverBindings[i], &gEfiDriverBindingProtocolGuid, (VOID**)&Drv);
    if (EFI_ERROR(Status) || Drv == NULL) {
      continue;
    }

    // Call Supported()
    Status = Drv->Supported (Drv, ControllerHandle, NULL);
    Print (L"  DriverBinding handle %p Supported() => 0x%lx\n", DriverBindings[i], Status);

    if (Status == EFI_SUCCESS) {
      // Print LoadedImage (if available)
      EFI_LOADED_IMAGE_PROTOCOL *Li = NULL;
      Status = gBS->HandleProtocol (DriverBindings[i], &gEfiLoadedImageProtocolGuid, (VOID**)&Li);
      if (!EFI_ERROR(Status) && Li != NULL) {
        CHAR16 *s = NULL;
        if (Li->FilePath != NULL) {
          s = MyDevicePathToStr (Li->FilePath);
        }
        Print (L"    Driver image: %p FilePath=%s\n", Li->ImageBase, s ? s : L"(no path)");
        if (s) FreePool (s);
      } else {
        Print (L"    (No LoadedImage on driver-binding handle) status=0x%lx\n", Status);
      }

      // Attempt Start(); capture return code
      Status = Drv->Start (Drv, ControllerHandle, NULL);
      Print (L"    DriverBinding->Start() returned 0x%lx\n", Status);

      // If Start succeeded, optionally stop it to keep system stable
      if (!EFI_ERROR(Status)) {
        Print (L"    -> Start succeeded; attempting Stop() to unbind for cleanliness.\n");
        EFI_STATUS St = Drv->Stop (Drv, ControllerHandle, 0, NULL);
        Print (L"    -> Stop() returned 0x%lx\n", St);
      }
    }
  }

  if (DriverBindings) FreePool (DriverBindings);
  Print (L"--- End trying Start() ---\n");
}



////////////////////////////////////////////////////////////// this is part strice for Virtual Box //////////////////////////////////////////////////////////////////
// --------------------------- Soft-reset helper (optional, gated) ---------------------------
// Set to 1 to actually perform MMIO writes that reset ports. Keep 0 to do read-only checks.
#ifndef ENABLE_PORT_RESET
#define ENABLE_PORT_RESET 1
#endif

#define EHCI_PORTSC_OFFSET  0x44   // common EHCI offset for PORTSC (OpBase + 0x44 + 4*(port-1))
#define EHCI_USBCMD_OFFSET  0x00
#define EHCI_USBCMD_RUN_BIT (1u << 0)
#define EHCI_PORT_RESET_BIT (1u << 8)     // typical EHCI Port Reset (PR) bit position
#define EHCI_PORT_PE_BIT    (1u << 2)     // Port Enable

// OHCI common registers (offsets are the commonly used values in many OHCI implementations).
// NOTE: OHCI root hub port status registers may appear at 0x44 + 4*(port-1) on many implementations,
// but the OHCI register map uses HcRhPortStatus[] starting typically at 0x44 as well.
#define OHCI_HC_REVISION_OFFSET       0x00
#define OHCI_HC_CONTROL_OFFSET        0x04
#define OHCI_HC_CMDSTATUS_OFFSET      0x08
#define OHCI_HC_HCCA_OFFSET           0x0C
#define OHCI_HC_RH_DESCRIPTOR_A       0x30
#define OHCI_HC_RH_DESCRIPTOR_B       0x34
#define OHCI_HC_RH_STATUS             0x38
#define OHCI_HC_RH_PORT_STATUS_BASE   0x44   // HcRhPortStatus[0] often at 0x44; port0 = offset 0x44

#define OHCI_RHPS_PORT_RESET_BIT      (1u << 8)  // typical bit for RootHub PortReset in OHCI RHPS
#define OHCI_RHPS_CCS_BIT             (1u << 0)  // Current Connect Status
#define OHCI_RHPS_PES_BIT             (1u << 1)  // Port Enable Status (read)
#define OHCI_RHPS_OVERCUR_CHG         (1u << 3)
#define OHCI_RHPS_CONNECT_STATUS_CHG  (1u << 0)  // sometimes same bit semantics on write

// Timeout settings (microseconds)
#define PORT_RESET_TIMEOUT_MS 1000
#define PORT_RESET_POLL_MS    50

// Utility: safe write (only if ENABLE_PORT_RESET). Always print action taken.
STATIC
EFI_STATUS
SafeWriteOp32 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               Offset,
  IN UINT32              Value
  )
{
#if ENABLE_PORT_RESET
  EFI_STATUS st;
  st = PciIo->Mem.Write (PciIo, EfiPciIoWidthUint32, BarIndex, Offset, 1, &Value);
  if (!EFI_ERROR(st)) {
    Print (L"    Wrote [BAR%u+0x%03x] = 0x%08x\n", BarIndex, (UINT32)Offset, Value);
  } else {
    Print (L"    Write to [BAR%u+0x%03x] failed: 0x%lx\n", BarIndex, (UINT32)Offset, st);
  }
  return st;
#else
  // Print what we *would* do, but don't write.
  Print (L"    (DRY-RUN) Would write [BAR%u+0x%03x] = 0x%08x (ENABLE_PORT_RESET=0)\n", BarIndex, (UINT32)Offset, Value);
  return EFI_SUCCESS;
#endif
}

// Utility: safe read (wraps PciIo->Mem.Read with printing)
STATIC
EFI_STATUS
SafeReadOp32 (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               Offset,
  OUT UINT32             *Value
  )
{
  EFI_STATUS st;
  st = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, Offset, 1, Value);
  if (EFI_ERROR(st)) {
    Print (L"    Read [BAR%u+0x%03x] failed: 0x%lx\n", BarIndex, (UINT32)Offset, st);
  } else {
    Print (L"    Read [BAR%u+0x%03x] => 0x%08x\n", BarIndex, (UINT32)Offset, *Value);
  }
  return st;
}

// Try EHCI port soft-reset (best-effort).
// PciIo already mapped, BarIndex and OpBase given (OpBase = CapLength usually).
STATIC
VOID
EhciSoftResetPort (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               OpBase,
  IN UINT32              PortIndex
  )
{
  UINTN PortOffset = OpBase + EHCI_PORTSC_OFFSET + (PortIndex - 1) * 4;
  UINT32 PortSc = 0;
  EFI_STATUS st;

  Print (L"  [EHCI] port%u: reading PORTSC at offset 0x%03x\n", (UINT32)PortIndex, (UINT32)PortOffset);
  st = SafeReadOp32 (PciIo, BarIndex, PortOffset, &PortSc);
  if (EFI_ERROR(st)) return;

  UINTN CCS = (PortSc >> 0) & 1;
  if (!CCS) {
    Print (L"    Port%u no device connected - skipping reset.\n", (UINT32)PortIndex);
    return;
  }

  // Set Port Reset bit (typical EHCI PR bit at bit 8). Write a 1 and poll until PE=1 or timeout.
  UINT32 WriteVal = PortSc | EHCI_PORT_RESET_BIT;
#if ENABLE_PORT_RESET
  Print (L"  [EHCI] Port%u: initiating port reset (writing PR bit)\n", (UINT32)PortIndex);
#endif
  SafeWriteOp32 (PciIo, BarIndex, PortOffset, WriteVal);

  // Poll until PortEnable bit is set (PE bit) or timeout.
  UINTN TimeoutLoops = (PORT_RESET_TIMEOUT_MS / PORT_RESET_POLL_MS);
  for (UINTN k = 0; k < TimeoutLoops; ++k) {
    gBS->Stall (PORT_RESET_POLL_MS * 1000);
    st = SafeReadOp32 (PciIo, BarIndex, PortOffset, &PortSc);
    if (EFI_ERROR(st)) break;
    if ((PortSc & EHCI_PORT_PE_BIT) != 0) {
      Print (L"    Port%u enabled (PE=1) after reset.\n", (UINT32)PortIndex);
      return;
    }
  }
  Print (L"    Port%u did not become enabled within timeout.\n", (UINT32)PortIndex);
}

// Try OHCI root-hub port soft-reset (best-effort)
// For many OHCI implementations the RH port status registers are at OpBase + OHCI_HC_RH_PORT_STATUS_BASE + 4*(port-1)
STATIC
VOID
OhciSoftResetPort (
  IN EFI_PCI_IO_PROTOCOL *PciIo,
  IN UINT8               BarIndex,
  IN UINTN               OpBase,
  IN UINT32              PortIndex
  )
{
  UINTN PortOffset = OpBase + OHCI_HC_RH_PORT_STATUS_BASE + (PortIndex - 1) * 4;
  UINT32 Rhps = 0;
  EFI_STATUS st;

  Print (L"  [OHCI] port%u: reading RHPS at offset 0x%03x\n", (UINT32)PortIndex, (UINT32)PortOffset);
  st = SafeReadOp32 (PciIo, BarIndex, PortOffset, &Rhps);
  if (EFI_ERROR(st)) return;

  UINTN CCS = (Rhps >> 0) & 1;
  if (!CCS) {
    Print (L"    Port%u no device connected - skipping OHCI reset.\n", (UINT32)PortIndex);
    return;
  }

  // For OHCI Root Hub Port Status, setting the Port Reset bit (often bit 8) by writing 1 triggers reset.
  UINT32 WriteVal = Rhps | OHCI_RHPS_PORT_RESET_BIT;
#if ENABLE_PORT_RESET
  Print (L"  [OHCI] Port%u: initiating RH port reset (writing PR bit)\n", (UINT32)PortIndex);
#endif
  SafeWriteOp32 (PciIo, BarIndex, PortOffset, WriteVal);

  // Poll until PES (Port Enable Status) becomes 1 or timeout
  UINTN TimeoutLoops = (PORT_RESET_TIMEOUT_MS / PORT_RESET_POLL_MS);
  for (UINTN k = 0; k < TimeoutLoops; ++k) {
    gBS->Stall (PORT_RESET_POLL_MS * 1000);
    st = SafeReadOp32 (PciIo, BarIndex, PortOffset, &Rhps);
    if (EFI_ERROR(st)) break;
    if ((Rhps & OHCI_RHPS_PES_BIT) != 0) {
      Print (L"    Port%u PES=1 after reset; port enabled.\n", (UINT32)PortIndex);
      return;
    }
  }
  Print (L"    Port%u did not become enabled within timeout (OHCI)\n", (UINT32)PortIndex);
}

// Top-level: detect controller type and optionally reset specified ports.
// Ports[] is an array of port numbers (e.g., {1,2}) with Count ports.
STATIC
VOID
AutoProbeAndMaybeResetController (
  IN UINT16 VendorId,
  IN UINT16 DeviceId,
  IN UINT32 *Ports,
  IN UINTN  Count
  )
{
  EFI_PCI_IO_PROTOCOL *PciIo = NULL;
  EFI_HANDLE PciH = FindPciHandleByVendorDevice (VendorId, DeviceId, &PciIo);
  if (PciH == NULL || PciIo == NULL) {
    Print (L"AutoProbe: PCI %04x:%04x not found\n", VendorId, DeviceId);
    return;
  }

  UINTN Seg, Bus, Dev, Func;
  PciIo->GetLocation (PciIo, &Seg, &Bus, &Dev, &Func);
  Print (L"\nAutoProbe: PCI %02x:%02x.%x handle=%p\n", (UINT32)Bus, (UINT32)Dev, (UINT32)Func, PciH);

  // read class/progif
  UINT32 ClassDword = 0;
  PciIo->Pci.Read (PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassDword);
  UINT8 ClassCode = (ClassDword >> 24) & 0xFF;
  UINT8 SubClass  = (ClassDword >> 16) & 0xFF;
  UINT8 ProgIf    = (ClassDword >> 8)  & 0xFF;
  Print (L"  PCI class=0x%02x subclass=0x%02x progIf=0x%02x\n", ClassCode, SubClass, ProgIf);

  // find CAP region / opbase like before:
  BOOLEAN Found = FALSE;
  for (UINT8 BarIndex = 0; BarIndex < MAX_BARS && !Found; ++BarIndex) {
    UINT8 CapLen = 0xFF;
    EFI_STATUS st = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLen);
    if (EFI_ERROR(st)) continue;
    if (CapLen == 0xFF || CapLen == 0x00) continue;

    Found = TRUE;
    UINTN OpBase = (UINTN)CapLen;
    Print (L"  Found candidate BAR%u CapLen/OpBase=0x%02x\n", BarIndex, (UINT32)OpBase);

    // Basic read of a few control registers
    UINT32 v;
    SafeReadOp32 (PciIo, BarIndex, OpBase + 0x00, &v); // USBCMD or HcRevision
    SafeReadOp32 (PciIo, BarIndex, OpBase + 0x04, &v); // USBSTS or HcControl

    // For EHCI (ProgIf 0x20) use EHCI reset routine
    if (ProgIf == 0x20) {
      Print (L"  Controller appears to be EHCI (ProgIf=0x20). Running EHCI port probe/reset on provided ports.\n");
      for (UINTN i = 0; i < Count; ++i) {
        EhciSoftResetPort (PciIo, BarIndex, OpBase, Ports[i]);
      }
    } else if (ProgIf == 0x10) {
      // OHCI
      Print (L"  Controller appears to be OHCI (ProgIf=0x10). Running OHCI port probe/reset on provided ports.\n");
      for (UINTN i = 0; i < Count; ++i) {
        OhciSoftResetPort (PciIo, BarIndex, OpBase, Ports[i]);
      }
    } else {
      Print (L"  Unknown ProgIf=0x%02x — not EHCI/OHCI. No reset attempted.\n", ProgIf);
    }

    // stop after first candidate
    break;
  }

  if (!Found) {
    Print (L"  No accessible MMIO BAR found for this controller (skipped).\n");
  }

  Print (L"AutoProbe: done for %04x:%04x\n\n", VendorId, DeviceId);
}
////////////////////////////////////////////////////////////// this is part strice for Virtual Box //////////////////////////////////////////////////////////////////


// OHCI register dump: read-only
// Adds: DumpOhciRegistersForPciHandle(VendorId, DeviceId)
//
// Requires these from your codebase:
//   - FindPciHandleByVendorDevice(UINT16 Vendor, UINT16 Device, EFI_PCI_IO_PROTOCOL **PciIoOut)
//   - MAX_BARS, MAX_PORTS_SAFE (already in your project)
//   - gBS, Print, EFI types and Protocol GUIDs included already in your file
//
STATIC
EFI_STATUS
DumpOhciRegistersForPciHandle (
  IN UINT16 VendorId,
  IN UINT16 DeviceId
  )
{
  EFI_HANDLE       PciHandle;
  EFI_PCI_IO_PROTOCOL *PciIo = NULL;
  EFI_STATUS       Status;
  UINT8            CapLen;
  BOOLEAN          Found = FALSE;

  Print (L"\n--- OHCI register dump for PCI %04x:%04x ---\n", VendorId, DeviceId);

  PciHandle = FindPciHandleByVendorDevice (VendorId, DeviceId, &PciIo);
  if (PciHandle == NULL || PciIo == NULL) {
    Print (L"  PCI device %04x:%04x not found or no PciIo\n", VendorId, DeviceId);
    return EFI_NOT_FOUND;
  }

  // Probe BARs to find an MMIO region with a non-0xFF first byte (cap length)
  for (UINT8 BarIndex = 0; BarIndex < MAX_BARS; ++BarIndex) {
    CapLen = 0xFF;
    Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint8, BarIndex, 0x00, 1, &CapLen);
    if (EFI_ERROR (Status)) {
      continue;
    }
    if (CapLen == 0xFF || CapLen == 0x00) {
      continue;
    }

    Found = TRUE;
    UINTN OpBase = (UINTN)CapLen;
    Print (L"  Using BAR%u OpBase=0x%03x\n", (UINT32)BarIndex, (UINT32)OpBase);

    // helper to read op-reg and print
    #define READ_OP_REG32(off, var) \
      do { \
        Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, (off) + OpBase, 1, &(var)); \
        if (EFI_ERROR (Status)) { \
          Print (L"    Read [BAR%u+0x%03x] failed: 0x%lx\n", (UINT32)BarIndex, (UINT32)((off)+OpBase), Status); \
        } else { \
          Print (L"    Read [BAR%u+0x%03x] => 0x%08x\n", (UINT32)BarIndex, (UINT32)((off)+OpBase), (UINT32)(var)); \
        } \
      } while (0)

    UINT32 HcRevision = 0;
    UINT32 HcControl = 0;
    UINT32 HcCommandStatus = 0;
    UINT32 HcHCCA = 0;
    UINT32 RhDescriptorA = 0;
    UINT32 RhDescriptorB = 0;
    UINT32 RhStatus = 0;

    READ_OP_REG32 (0x00, HcRevision);        // HcRevision (0x00)
    READ_OP_REG32 (0x04, HcControl);         // HcControl (0x04)
    READ_OP_REG32 (0x08, HcCommandStatus);   // HcCommandStatus (0x08)
    READ_OP_REG32 (0x0C, HcHCCA);            // HcHCCA (0x0C)
    // skip reserved gaps; root-hub descriptor offsets at 0x30..0x38
    READ_OP_REG32 (0x30, RhDescriptorA);     // HcRhDescriptorA (0x30)
    READ_OP_REG32 (0x34, RhDescriptorB);     // HcRhDescriptorB (0x34)
    READ_OP_REG32 (0x38, RhStatus);          // HcRhStatus    (0x38)

    // Determine number of root-hub ports from RhDescriptorA (OHCI: lower 8 bits = NbrPorts)
    UINT32 NPorts = (UINT32)(RhDescriptorA & 0xFF);
    if (NPorts == 0 || NPorts > MAX_PORTS_SAFE) {
      // fallback: cap to MAX_PORTS_SAFE to protect scanning
      Print (L"    Warning: RhDescriptorA reports NPorts=%u. Using safe cap %u\n", NPorts, MAX_PORTS_SAFE);
      NPorts = MAX_PORTS_SAFE;
    }
    Print (L"  Root Hub: HcRevision=0x%08x HcControl=0x%08x HcCmdSts=0x%08x HcHCCA=0x%08x\n",
           HcRevision, HcControl, HcCommandStatus, HcHCCA);
    Print (L"           RhDescriptorA=0x%08x RhDescriptorB=0x%08x RhStatus=0x%08x NPorts=%u\n",
           RhDescriptorA, RhDescriptorB, RhStatus, NPorts);

    // Read each root-hub port status (HcRhPortStatus[0] at OpBase + 0x44)
    for (UINT32 p = 1; p <= NPorts; ++p) {
      UINT32 PortReg = 0;
      UINTN PortOffset = OpBase + 0x44 + (p - 1) * 4;
      Status = PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, BarIndex, PortOffset, 1, &PortReg);
      if (EFI_ERROR (Status)) {
        Print (L"    Read PORT%u [BAR%u+0x%03x] failed: 0x%lx  (stopping port loop)\n", (UINT32)p, (UINT32)BarIndex, (UINT32)PortOffset, Status);
        break;
      }

      // conservative decode: Current Connect Status (CCS) usually bit0, Port Enable Status (PES) usually bit1, Port Reset often bit8.
      UINTN CCS = (PortReg >> 0) & 1;
      UINTN PES = (PortReg >> 1) & 1;
      UINTN PR  = (PortReg >> 8) & 1; // PortReset bit (writeable on many controllers)
      Print (L"    PORT%u: RHPS raw=0x%08x  (CCS=%u PES=%u PR=%u)\n", (UINT32)p, PortReg, (UINT32)CCS, (UINT32)PES, (UINT32)PR);
    }

    // done with first candidate BAR
    break;
  } // for BarIndex

  if (!Found) {
    Print (L"  No accessible MMIO BAR found for this PCI device\n");
    return EFI_NOT_FOUND;
  }

  Print (L"--- End OHCI dump ---\n\n");
  return EFI_SUCCESS;
}

STATIC
VOID
DumpPciConfigSummary (
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
  EFI_STATUS St;
  UINT32 D0;
  // Read DWORDs from config space 0x00..0x3C
  St = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x00, 1, &D0);
  if (EFI_ERROR(St)) {
    Print(L"  Pci.Read cfg[0x00] failed: %r\n", St);
    return;
  }
  Print(L"  PCI cfg[0x00] = 0x%08x\n", (UINT32)D0);
  UINT32 Cmd;
  PciIo->Pci.Read(PciIo, EfiPciIoWidthUint16, 0x04, 1, &Cmd);
  Print(L"  PCI CMD (0x04)=0x%04x  (bit0 I/O, bit1 MEM, bit2 BusMaster)\n", (UINT32)Cmd);
  UINT32 Bar0;
  PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x10, 1, &Bar0);
  Print(L"  PCI BAR0 (cfg 0x10)=0x%08x\n", (UINT32)Bar0);
}



// UEFI MAIN /////////////////////
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

  // Additional mapped checks:
  MapUsb2HcToPciBdf ();

  // EHCI register dump (read-only)
  ScanAllPciForEhciAndCheck ();
  
		// anothers
		//ListHandlesAndProtocols ();
		ExamineUsbPortEnumerationAndCompanionOwnership();
		
		CheckUnenumeratedPortsAndDiagnose();
		
		gBS->Stall (200 * 1000);
		
		InspectPciAndAttachedUsbDevices (0x106B, 0x003F);
		
		gBS->Stall (200 * 1000);

  // NEW: inspect the specific PCI device 106B:003F and published USB handles
  EFI_PCI_IO_PROTOCOL *TmpPciIo = NULL;
  EFI_HANDLE PciH = FindPciHandleByVendorDevice(0x106B, 0x003F, &TmpPciIo);
  if (PciH != NULL) {
    Print(L"\nInspecting PCI device 106B:003F (handle=%p)\n", PciH);

    // show published UsbIo handles and their ports
    ListAllUsbIoHandlesWithPortNodes ();

    // try to locate driver-binding that supports this controller and dump/start it
    FindDriverBindingThatSupportsControllerAndDump (PciH);

    // small pause to allow driver to finish Start() if it succeeded
    gBS->Stall (200 * 1000);

    // re-run the diagnostic for the port(s) on this controller (optional)
    // you need to compute barindex/opbase as you already do elsewhere in your code.
    // Example:
    // DiagnosePortOnController (TmpPciIo, (UINT8)BarIndex, OpBase, 1);
  } else {
    Print(L"PCI device 106B:003F not found on this system\n");
  }
	
	
	/////////////////////////////////////////////////////////// code added 17-10-2025 ///////////////////////////////////////////////////////////////
	
	  // after your existing sequence and stalls
  CheckControllerPortHealthByVendorDevice (0x106B, 0x003F);

  // continue with existing flow
  CheckUsbDevices ();
  CheckUsbBlockIoAndReadLba0 ();


	
	// examine 
	/*
	This will print per-port summary and run DoDeviceLevelProbe for any published devices on ports 1 and 2. 
	For the pendrive on port 1 you should see the same INQUIRY behavior you already observed.
	For the Bluetooth device on port 2 you should see device/config/interface descriptors and a small safe IN test if an appropriate IN endpoint exists.
	*/
	  // after the existing sequence and stalls...
  ProbeControllerPortsSimple (0x106B, 0x003F, 1, 2);

	// stall for a momennt again
	gBS->Stall (200 * 1000);
	
	// next diagnose
	if (PciH != NULL) {
	
		  // show protocols currently on the controller handle
	  DumpProtocolsOnHandle (PciH);

	  // list all loaded DXE image device paths so you can find the module to inspect
	  ListLoadedImages ();

	  // attempt to start every driver that reports Supported()==EFI_SUCCESS
	  TryStartAllSupportingDrivers (PciH);

	  // re-run your health probe to see if anything changed
	  CheckControllerPortHealthByVendorDevice (0x106B, 0x003F);

		
	}
	
	// Virtual box stuff only
	  UINT32 ports_to_try[] = { 1, 2 };
	AutoProbeAndMaybeResetController (0x106B, 0x003F, ports_to_try, sizeof(ports_to_try)/sizeof(ports_to_try[0]));

	// After AutoProbeAndMaybeResetController(...)
	// small delay to let controller settle
	gBS->Stall(200 * 1000);

	// Try to start drivers that supported controller (you already have this helper)
	TryStartAllSupportingDrivers(PciH);

	// Also try ConnectController to let the driver manager bind any matching drivers
	EFI_STATUS St = gBS->ConnectController(PciH, NULL, NULL, TRUE);
	Print(L"gBS->ConnectController(PciH,NULL,NULL,TRUE) returned 0x%lx\n", St);

	// Give drivers a short time to enumerate children
	gBS->Stall(300 * 1000);

	// Rescan UsbIo handles
	ListAllUsbIoHandlesWithPortNodes();



	///////////////////////////////////// after virtual box stuff ////////////////////////////////////
	// chcialem sprawdzic pewne rzeczy tylko konkretnie pod Virtual Box ale nic to nie dziale wiec ide dalej
	
	// Example call in UefiMain after you find the PCI device:
	DumpOhciRegistersForPciHandle (0x106B, 0x003F);


	// try to connect any driver now:
	// probablyu PciH as handle on first arg
	//St = gBS->ConnectController(PciHandle, NULL, NULL, TRUE);
	St = gBS->ConnectController(PciH, NULL, NULL, TRUE);
	Print(L"ConnectController(PciHandle,NULL,NULL,TRUE) => %r\n", St);
	gBS->Stall(300 * 1000);
	ListAllUsbIoHandlesWithPortNodes();




	//////////////////////////////////////////////////////////// I SKIPED THIS PART -> log looks like /////////////////////////////////////////////////
	
										// Stage 3: find and exercise EFI_USB_IO_PROTOCOL devices
									  //CheckUsbDevices ();

									  // BlockIo checks and safe LBA0 read
									  //CheckUsbBlockIoAndReadLba0 ();
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	
	--- Stage 3: Locate EFI_USB_IO_PROTOCOL handles (actual devices) ---
Found 1 USB device handles (EFI_USB_I
--- Device 0 ---
UsbGetDeviceDescriptor: Status = 0x0
  bLength=18 bDescriptorType=1 bcdUSB=0x0200 idVendor=0x058F idProduct=0x6387 bDeviceClass=0x00
UsbGetConfigDescriptor: Status = 0x0
  wTotalLength=32 bNumInterfaces=1 bConfigurationValue=1
 skipping DumpHex 32
UsbGetInterfaceDescriptor: Status = 0x0
  Interface: bInterfaceNumber=0 bAlternateSetting=0 bInterfaceClass=0x08 bInterfaceSubClass=0x06 bInterfaceProtocol=0x50
  Endpoint 0: bEndpointAddress=0x01 bmAttributes=0x02 wMaxPacketSize=512 bInterval=0
  Endpoint 1: bEndpointAddress=0x82 bmAttributes=0x02 wMaxPacketSize=512 bInterval=0
  End of endpoint list (index 2). Last status=Not Found
UsbControlTransfer(GET_STATUS): Status = 0x0
  GET_STATUS returned UsbStatus=0x0 Data: 0000: 00 00
UsbPortReset (note: this may re-enumerate device): Status = 0x0
--- End of device 0 checks ---
--- Stage: Check BlockIo on USB mass-storage devices ---
  BlockIo handle 0: MediaId=0 BlockSize=512 LastBlock=125829119 MediaPresent=1 Removable=0
    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: 33 C0 8E D0 BC 00 7C 8E C0 8E D8 BE 00 7C BF 00
0010: 06 B9 00 02 FC F3 A4 50 68 1C 06 CB FB B9 04 00
0020: BD BE 07 80 7E 00 00 7C 0B 0F 85 0E 01 83 C5 10
0030: E2 F1 CD 18 88 56 00 55 C6 46 11 05 C6 46 10 00
    MBR signature 0x55AA found at offset 0x1FE
  BlockIo handle 1: MediaId=1 BlockSize=2048 LastBlock=2335935 MediaPresent=1 Removable=1
    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0010: 00 00 00 00 00 00 00 00 00 00 0000 00 00 00
0020: 00 00 00 00 00 00 00 00 00 00 000 00 00 00 00
0030: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    No MBR signature at 0x1FE (bytes=0x00 0x00)
  BlockIo handle 2: MediaId=1 BlockS    ReadBlocks(LBA0) returned 0x0
0000: FA 33 C0 8E D0 BC 00 7C FB 8C C8 8E D8 52 B4 0F
0010: CD 10 24 7F 3C 03 74 06 B4 00 B0 03 CD 10 E8 00
0020: 00 5E 81 EE 21 00 74 16 81 FE 00 7C 0F 85 7D 00
0030: 8C C8 3D 00 00 0F 85 84 00 EA 45 00 C0 07 8C C8
    No MBR signature at 0x1FE (bytes  BlockIo handle 3: MediaId=1 BlockSize=2048 LastBlock=2879 MediaPresent=1 Removable=1
    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: EB 3C 90 4D 53 44 4F 53 35 2E 30 00 02 01 01 00
0010: 02 E0 00 40 0B F0 09 00 12 00 02 00 00 00 00 00
0020: 00 00 00 00 00 00 29 78 48 A0 00 45 46 49 53 45
0030: 43 54 4F 52 20 20 46 41 54 31 32 20 20 20 FA 33
    MBR signature 0x55AA found at offset 0x1FE
  BlockIo handle 4: MediaId=1 BlockSize=2048 LastBlock=2335903 MediaPresent=1 Removable=1
    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: 01 00 02 00 5E 00 00 00 3C 0E F0 01 20 00 00 00
0010: 00 00 00 00 00 00 00 00 08 45 53 44 2D 49 53 4F
0020: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0030: 00 00 00 00 00 00 00 08 01 00 01 00 02 00 03 00
    No MBR signature at 0x1FE (bytes=0x00 0x00)
  BlockIo handle 5: MediaId=0 BlockSi    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: EB 58 90 4D 53 44 4F 53 35 2E 30 00 02 02 FE 19
0010: 02 00 00 00 00 F8 00 00 3F 00 FF 00 00 08 00 00
0020: 00 20 03 00 01 03 00 00 00 00 00 00 02 00 00 00
0030: 01 00 06 00 00 00 00 00 00 00 00 00 00 00 00 00
    MBR signature 0x55AA found at offset 0x1FE
  BlockIo handle 6: MediaId=0 BlockS    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0010: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0020: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0030: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    No MBR signature at 0x1FE (bytes=0x00 0x00)
  BlockIo handle 7: MediaId=0 BlockSize=512 LastBlock=124499022 MediaPresent=1 Removable=0
    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: EB 52 90 4E 54 46 53 20 20 20 20 00 02 08 00 00
0010: 00 00 00 00 00 F8 00 00 3F 00 FF 00 00 A8 03 00
0020: 00 00 00 00 80 00 80 00 4E B4 6B 07 00 00 00 00
0030: 00 00 0C 00 00 00 00 00 02 00 00 00 00 00 00 00
    MBR signature 0x55AA found at offset 0x1FE
  BlockIo handle 8: MediaId=0 BlockSize=512 LastBlock=1    ReadBlocks(LBA0) returned 0x0
    LBA0 (first 64 bytes) (len=64):
0000: EB 52 90 4E 54 46 53 20 20 20 20 00 02 08 00 00
0010: 00 00 00 00 00 F8 00 00 3F 00 FF 00 00 60 6F 07
0020: 00 00 00 00 80 00 80 00 FF 8F 100 00 00 00 00
0030: AA B0 00 00 00 00 00 00 02 00 00 00 00 00 00 00
    MBR signature 0x55AA found at offset 0x1FE
  BlockIo handle 9: MediaId=2 BlockSize=512 LastBlock=8165375 MediaPresent=1 Removable=1
    ReadBlocks(LBA0) returned 0x80000UsbDiagApp: Completed checks. You can re-run to captuDone. Press any key to exit...
	*/

  // Stage 3: find and exercise EFI_USB_IO_PROTOCOL devices
  //CheckUsbDevices ();

  // BlockIo checks and safe LBA0 read
  //CheckUsbBlockIoAndReadLba0 ();

  Print (L"UsbDiagApp: Completed checks. You can re-run to capture new events.\n");

  // wait for key
  Print(L"Done. Press any key to exit...\n");
  EFI_INPUT_KEY Key;
  while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) != EFI_SUCCESS) {
    // spin
  }

  return EFI_SUCCESS;
}
