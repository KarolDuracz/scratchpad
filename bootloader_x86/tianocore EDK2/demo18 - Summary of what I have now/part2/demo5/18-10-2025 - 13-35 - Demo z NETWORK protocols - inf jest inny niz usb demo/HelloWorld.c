/** @file
  HelloNet.c - UEFI app to enumerate network & bluetooth handles,
  print OpenProtocolInformation, device path and PCI info.

  - NO ConvertDevicePathToText() usage.
  - Custom DevicePathToSimpleText() implemented.
  - Local GUIDs declared for WiFi/Bluetooth/VLAN to avoid missing externs.
  - Read-only (no writes/resets).
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/PrintLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/BaseLib.h>
#include <Library/DebugLib.h>

#include <Protocol/SimpleNetwork.h>
#include <Protocol/ManagedNetwork.h>
#include <Protocol/PciIo.h>
#include <Protocol/DevicePath.h>

//
// Local GUIDs for protocols that may not have gEfi... externs in every EDK2 tree.
// We intentionally use different names to avoid colliding with any existing gEfi... globals.
//
STATIC EFI_GUID gWirelessMacConnGuid = {
  0x0DA55BC9, 0x45F8, 0x4BB4, {0x87,0x19,0x52,0x24,0xF1,0x8A,0x4D,0x45}
};

STATIC EFI_GUID gWirelessMacConn2Guid = {
  0x1B0FB9BF, 0x699D, 0x4FDD, {0xA7,0xC3,0x25,0x46,0x68,0x1B,0xF6,0x3B}
};

STATIC EFI_GUID gBluetoothIoGuid = {
  0x467313DE, 0x4E30, 0x43F1, {0x94,0x3E,0x32,0x3F,0x89,0x84,0x5D,0xB5}
};

STATIC EFI_GUID gBluetoothHcGuid = {
  0xB3930571, 0xBEBA, 0x4FC5, {0x92,0x03,0x94,0x27,0x24,0x2E,0x6A,0x43}
};

STATIC EFI_GUID gVlanConfigGuid = {
  0x9E23D768, 0xD2F3, 0x4366, {0x9F,0xC3,0x3A,0x7A,0xBA,0x86,0x43,0x74}
};

//
// Small helper to append formatted text into a CHAR16 buffer.
// Returns number of CHAR16 characters written (excluding NUL).
//
STATIC
UINTN
AppendUnicode (
  IN OUT CHAR16 *Buffer,
  IN     UINTN   BufferSize, // total length in CHAR16 units
  IN     UINTN   Offset,     // current offset in CHAR16 units
  IN     CONST CHAR16 *Format,
  ...
  )
{
  VA_LIST Args;
  UINTN Rem;
  UINTN Written;

  if (Offset >= BufferSize) {
    return 0;
  }

  Rem = BufferSize - Offset;

  VA_START (Args, Format);
  // UnicodeVSPrint expects size in bytes.
  UnicodeVSPrint (Buffer + Offset, Rem * sizeof (CHAR16), Format, Args);
  VA_END (Args);

  Written = StrLen (Buffer + Offset);
  return Written;
}

//
// Convert a device path to a compact but useful text representation.
// Returns allocated CHAR16* or NULL on alloc failure. Caller must FreePool().
//
STATIC
CHAR16 *
DevicePathToSimpleText (
  IN EFI_DEVICE_PATH_PROTOCOL *DevPath
  )
{
  if (DevPath == NULL) {
    CHAR16 *NullStr = AllocatePool (sizeof (CHAR16) * 24);
    if (NullStr) {
      UnicodeSPrint (NullStr, 24 * sizeof (CHAR16), L"<NULL DevicePath>");
    }
    return NullStr;
  }

  CONST UINTN BufChars = 1024;
  CHAR16 *Buf = AllocatePool (BufChars * sizeof (CHAR16));
  if (Buf == NULL) {
    return NULL;
  }

  Buf[0] = L'\0';
  UINTN pos = 0;
  EFI_DEVICE_PATH_PROTOCOL *Node = DevPath;

  while (Node != NULL) {
    UINT8 *raw = (UINT8 *)Node;
    UINT16 Len = (UINT16)(raw[2] | (raw[3] << 8));
    if (Len < sizeof (EFI_DEVICE_PATH_PROTOCOL)) {
      pos += AppendUnicode (Buf, BufChars, pos, L"[BAD_NODE len=%u] ", Len);
      break;
    }

    // End node
    if (raw[0] == 0x7F && raw[1] == 0xFF) {
      pos += AppendUnicode (Buf, BufChars, pos, L"[END]");
      break;
    }

    pos += AppendUnicode (Buf, BufChars, pos, L"[%02x:%02x len=%u] ", raw[0], raw[1], Len);

    UINTN payloadLen = (Len > 4) ? (Len - 4) : 0;
    UINTN toShow = (payloadLen > 8) ? 8 : payloadLen;
    if (toShow > 0) {
      pos += AppendUnicode (Buf, BufChars, pos, L"(");
      for (UINTN i = 0; i < toShow; ++i) {
        pos += AppendUnicode (Buf, BufChars, pos, L"%02x", raw[4 + i]);
        if (i + 1 < toShow) {
          pos += AppendUnicode (Buf, BufChars, pos, L":");
        }
      }
      pos += AppendUnicode (Buf, BufChars, pos, L") ");
    }

    Node = (EFI_DEVICE_PATH_PROTOCOL *)((UINT8 *)Node + Len);
    if (Len == 0) {
      break;
    }
  }

  return Buf;
}

STATIC
VOID
PrintOpenAttributes (
  IN UINT64 Attributes
  )
{
  if (Attributes & EFI_OPEN_PROTOCOL_BY_HANDLE_PROTOCOL) {
    Print (L" BY_HANDLE_PROTOCOL");
  }
  if (Attributes & EFI_OPEN_PROTOCOL_BY_DRIVER) {
    Print (L" BY_DRIVER");
  }
  if (Attributes & EFI_OPEN_PROTOCOL_BY_CHILD_CONTROLLER) {
    Print (L" BY_CHILD_CONTROLLER");
  }
  if (Attributes & EFI_OPEN_PROTOCOL_GET_PROTOCOL) {
    Print (L" GET_PROTOCOL");
  }
  if (Attributes & EFI_OPEN_PROTOCOL_TEST_PROTOCOL) {
    Print (L" TEST_PROTOCOL");
  }
  if (Attributes & EFI_OPEN_PROTOCOL_EXCLUSIVE) {
    Print (L" EXCLUSIVE");
  }
  if (Attributes == 0) {
    Print (L" (0)");
  }
}

STATIC
VOID
DumpPciConfigIfPresent (
  IN EFI_HANDLE Handle
  )
{
  EFI_STATUS Status;
  EFI_PCI_IO_PROTOCOL *PciIo;

  Status = gBS->HandleProtocol (Handle, &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
  if (EFI_ERROR (Status) || PciIo == NULL) {
    return;
  }

  Print (L"  [PCI] EFI_PCI_IO_PROTOCOL present - reading first 64 bytes of config space\n");

  UINT8 Config[64];
  Status = PciIo->Pci.Read (PciIo, EfiPciIoWidthUint8, 0, sizeof (Config), Config);
  if (EFI_ERROR (Status)) {
    Print (L"    PCI config read failed: %r\n", Status);
    return;
  }

  UINT16 VendorId  = (UINT16)(Config[0] | (Config[1] << 8));
  UINT16 DeviceId  = (UINT16)(Config[2] | (Config[3] << 8));
  UINT8  ProgIF    = Config[9];
  UINT8  SubClass  = Config[10];
  UINT8  BaseClass = Config[11];

  Print (L"    VendorId: 0x%04x DeviceId: 0x%04x  Class: 0x%02x SubClass:0x%02x ProgIF:0x%02x\n",
         VendorId, DeviceId, BaseClass, SubClass, ProgIF);

  for (UINTN i = 0; i < 6; ++i) {
    UINT32 Bar = (UINT32)(Config[0x10 + i*4] |
                         (Config[0x10 + i*4 + 1] << 8) |
                         (Config[0x10 + i*4 + 2] << 16) |
                         (Config[0x10 + i*4 + 3] << 24));
    Print (L"    BAR%u: 0x%08x\n", (UINT32)i, Bar);
  }
}

STATIC
VOID
InspectHandlesByProtocol (
  IN EFI_GUID *ProtocolGuid,
  IN CHAR16   *ProtocolName,
  IN UINT8    SpecialCase
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *HandleBuffer = NULL;
  UINTN HandleCount = 0;

  Status = gBS->LocateHandleBuffer (ByProtocol, ProtocolGuid, NULL, &HandleCount, &HandleBuffer);
  if (EFI_ERROR (Status) || HandleCount == 0) {
    Print (L"%s: no handles found (%r)\n\n", ProtocolName, Status);
    return;
  }

  Print (L"%s: found %u handle(s)\n", ProtocolName, (UINT32)HandleCount);

  for (UINTN i = 0; i < HandleCount; ++i) {
    EFI_HANDLE H = HandleBuffer[i];
    Print (L" Handle[%u] = %p\n", (UINT32)i, H);

    // DevicePath -> custom string
    EFI_DEVICE_PATH_PROTOCOL *DevPath = NULL;
    Status = gBS->HandleProtocol (H, &gEfiDevicePathProtocolGuid, (VOID **)&DevPath);
    if (!EFI_ERROR (Status) && DevPath != NULL) {
      CHAR16 *DpTxt = DevicePathToSimpleText (DevPath);
      if (DpTxt != NULL) {
        Print (L"  DevicePath: %s\n", DpTxt);
        FreePool (DpTxt);
      } else {
        Print (L"  DevicePath: <alloc failed>\n");
      }
    } else {
      Print (L"  DevicePath: <none>\n");
    }

    // ProtocolsPerHandle: note correct pointer-level (EFI_GUID **)
    EFI_GUID **ProtArray = NULL;
    UINTN ProtCount = 0;
    Status = gBS->ProtocolsPerHandle (H, &ProtArray, &ProtCount);
    if (!EFI_ERROR (Status) && ProtCount > 0) {
      Print (L"  Protocols installed on handle (%u):\n", (UINT32)ProtCount);
      for (UINTN p = 0; p < ProtCount; ++p) {
        // ProtArray[p] is EFI_GUID*
        Print (L"   - %g\n", ProtArray[p]);
      }
      FreePool (ProtArray);
    } else {
      Print (L"  Protocols installed on handle: <none or error %r>\n", Status);
    }

    // OpenProtocolInformation
    EFI_OPEN_PROTOCOL_INFORMATION_ENTRY *EntryBuffer = NULL;
    UINTN EntryCount = 0;
    Status = gBS->OpenProtocolInformation (H, ProtocolGuid, &EntryBuffer, &EntryCount);
    if (!EFI_ERROR (Status) && EntryCount > 0) {
      Print (L"  OpenProtocolInformation (entries: %u):\n", (UINT32)EntryCount);
      for (UINTN e = 0; e < EntryCount; ++e) {
        Print (L"    Agent=%p Attr:", EntryBuffer[e].AgentHandle);
        PrintOpenAttributes (EntryBuffer[e].Attributes);
        Print (L"  OpenCount=%u\n", (UINT32)EntryBuffer[e].OpenCount);
      }
      FreePool (EntryBuffer);
    } else {
      Print (L"  OpenProtocolInformation: none or error (%r)\n", Status);
    }

    //
    // Special-cased, read-only introspection
    //
    if (SpecialCase == 1) { // SNP
      EFI_SIMPLE_NETWORK_PROTOCOL *Snp;
      Status = gBS->HandleProtocol (H, &gEfiSimpleNetworkProtocolGuid, (VOID **)&Snp);
      if (!EFI_ERROR (Status) && Snp != NULL) {
        EFI_SIMPLE_NETWORK_MODE *Mode = Snp->Mode;
        if (Mode != NULL) {
          Print (L"  SNP Mode: IfType=%u HwAddrLen=%u MediaPresent=%u\n",
                 Mode->IfType, Mode->HwAddressSize, Mode->MediaPresent);
          Print (L"  MAC: ");
          for (UINTN b = 0; b < Mode->HwAddressSize; ++b) {
            Print (L"%02x", Mode->CurrentAddress.Addr[b]);
            if (b + 1 < Mode->HwAddressSize) Print (L":");
          }
          Print (L"\n");
        } else {
          Print (L"  SNP Mode: <NULL>\n");
        }
      } else {
        Print (L"  SNP: HandleProtocol failed (%r)\n", Status);
      }
    } else if (SpecialCase == 2) { // MNP
      EFI_MANAGED_NETWORK_PROTOCOL *Mnp;
      Status = gBS->HandleProtocol (H, &gEfiManagedNetworkProtocolGuid, (VOID **)&Mnp);
      if (!EFI_ERROR (Status) && Mnp != NULL) {
        // Per UEFI spec: GetModeData(OUT EFI_MANAGED_NETWORK_CONFIG_DATA **Config OPTIONAL,
        //                             OUT EFI_SIMPLE_NETWORK_MODE **SnpMode OPTIONAL)
        EFI_MANAGED_NETWORK_CONFIG_DATA Cfg;
        EFI_SIMPLE_NETWORK_MODE SnpMode;

        if (Mnp->GetModeData != NULL) {
          Status = Mnp->GetModeData (Mnp, &Cfg, &SnpMode);
          if (!EFI_ERROR (Status)) {
            //if (SnpMode.HwAddressSize != NULL) {
              Print (L"  MNP SnpMode: IfType=%u HwAddrLen=%u MediaPresent=%u\n",
                     SnpMode.IfType, SnpMode.HwAddressSize, SnpMode.MediaPresent);
              Print (L"  MAC: ");
              for (UINTN b = 0; b < SnpMode.HwAddressSize; ++b) {
                Print (L"%02x", SnpMode.CurrentAddress.Addr[b]);
                if (b + 1 < SnpMode.HwAddressSize) Print (L":");
              }
              Print (L"\n");
            //} else {
            //  Print (L"  MNP GetModeData returned no SnpMode pointer\n");
            //}
            // Do NOT free Cfg or SnpMode here: implementations may return internal pointers.
          } else {
            Print (L"  MNP GetModeData failed: %r\n", Status);
          }
        } else {
          Print (L"  MNP: GetModeData not implemented by this instance\n");
        }
      } else {
        Print (L"  MNP: HandleProtocol failed (%r)\n", Status);
      }
    } else if (SpecialCase == 3) { // WiFi presence
      Print (L"  WiFi protocol handle present (no further introspection performed)\n");
    } else if (SpecialCase == 4) { // Bluetooth presence
      Print (L"  Bluetooth protocol handle present (no further introspection performed)\n");
    }

    // Dump PCI config if present
    DumpPciConfigIfPresent (H);

    Print (L"\n");
  }

  if (HandleBuffer) {
    FreePool (HandleBuffer);
  }
}

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  Print (L"HelloNet: network & bluetooth enumerator (read-only)\n\n");

  InspectHandlesByProtocol (&gEfiSimpleNetworkProtocolGuid, L"EFI_SIMPLE_NETWORK_PROTOCOL (SNP)", 1);
  InspectHandlesByProtocol (&gEfiManagedNetworkProtocolGuid, L"EFI_MANAGED_NETWORK_PROTOCOL (MNP)", 2);

  InspectHandlesByProtocol (&gWirelessMacConnGuid,  L"EFI_WIRELESS_MAC_CONNECTION_PROTOCOL (WiFi)",  3);
  InspectHandlesByProtocol (&gWirelessMacConn2Guid, L"EFI_WIRELESS_MAC_CONNECTION_II_PROTOCOL (WiFi2)", 3);

  InspectHandlesByProtocol (&gBluetoothHcGuid, L"EFI_BLUETOOTH_HC_PROTOCOL (Bluetooth HC)", 4);
  InspectHandlesByProtocol (&gBluetoothIoGuid, L"EFI_BLUETOOTH_IO_PROTOCOL (Bluetooth IO)", 4);

  InspectHandlesByProtocol (&gVlanConfigGuid, L"EFI_VLAN_CONFIG_PROTOCOL (VLAN)", 3);

  Print (L"Done.\n");
  return EFI_SUCCESS;
}
