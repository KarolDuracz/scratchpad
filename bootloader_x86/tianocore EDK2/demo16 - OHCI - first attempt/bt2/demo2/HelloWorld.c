/*
  USB topology tree printer (PCI -> root hub -> ports -> devices)
  - Finds PCI USB host controllers (Class 0x0C SubClass 0x03)
  - For each host, finds USB device handles whose DevicePath is a child of the PCI device's DevicePath
  - Parses textual DevicePath "USB(0xN,0xM)" nodes and uses the first value (port number)
  - Builds and prints a hierarchical tree of ports -> (hub/device)
*/

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/PrintLib.h>
#include <Protocol/PciIo.h>
#include <Protocol/DevicePath.h>
#include <Protocol/DevicePathToText.h>
#include <Protocol/UsbIo.h>
#include <IndustryStandard/Usb.h>

#ifndef END_DEVICE_PATH_TYPE
#define END_DEVICE_PATH_TYPE 0x7F
#endif
#ifndef END_ENTIRE_DEVICE_PATH_SUBTYPE
#define END_ENTIRE_DEVICE_PATH_SUBTYPE 0xFF
#endif

// -------------------- small helpers --------------------

STATIC
UINTN
GetDevicePathSizeBytes (
  IN EFI_DEVICE_PATH_PROTOCOL  *DevicePath
  )
{
  EFI_DEVICE_PATH_PROTOCOL *Node;
  UINTN Size = 0;

  if (DevicePath == NULL) {
    return 0;
  }

  Node = DevicePath;
  while (TRUE) {
    UINT8  *Raw = (UINT8 *)Node;
    UINT16 Len;

    Len = *(UINT16 *)(Raw + 2);
    if (Len < sizeof(EFI_DEVICE_PATH_PROTOCOL)) {
      return Size;
    }

    Size += Len;

    if (Node->Type == END_DEVICE_PATH_TYPE && Node->SubType == END_ENTIRE_DEVICE_PATH_SUBTYPE) {
      break;
    }

    Node = (EFI_DEVICE_PATH_PROTOCOL *)((UINT8 *)Node + Len);
  }

  return Size;
}

STATIC
BOOLEAN
DevicePathIsPrefixBinary (
  IN EFI_DEVICE_PATH_PROTOCOL *Prefix,
  IN UINTN                    PrefixSize,
  IN EFI_DEVICE_PATH_PROTOCOL *Path,
  IN UINTN                    PathSize
  )
{
  if (Prefix == NULL || Path == NULL) {
    return FALSE;
  }
  if (PrefixSize == 0 || PathSize < PrefixSize) {
    return FALSE;
  }

  return (CompareMem((VOID *)Prefix, (VOID *)Path, PrefixSize) == 0);
}

// trim leading spaces
STATIC
CHAR16 *
TrimLeft (
  CHAR16 *s
  )
{
  while (s != NULL && *s == L' ') ++s;
  return s;
}

// parse a single hex number (0x... or digits) from a string; advances pointer (by reference)
STATIC
BOOLEAN
ParseHexNumberAndAdvance(
  IN OUT CHAR16 **StrPtr,
  OUT UINT32 *Value
  )
{
  CHAR16 *s = *StrPtr;
  UINT32 val = 0;
  BOOLEAN any = FALSE;

  // skip spaces
  while (*s == L' ') ++s;

  // support optional 0x
  if (s[0] == L'0' && (s[1] == L'x' || s[1] == L'X')) {
    s += 2;
  }

  while (1) {
    CHAR16 c = *s;
    UINT8 d;
    if (c >= L'0' && c <= L'9') d = (UINT8)(c - L'0');
    else if (c >= L'a' && c <= L'f') d = (UINT8)(10 + c - L'a');
    else if (c >= L'A' && c <= L'F') d = (UINT8)(10 + c - L'A');
    else break;
    val = (val << 4) | d;
    any = TRUE;
    ++s;
  }

  if (!any) return FALSE;

  *Value = val;
  *StrPtr = s;
  return TRUE;
}

// find next occurrence of substring (case-sensitive) in wide string
STATIC
CHAR16 *
FindSubstring(
  IN CONST CHAR16 *Haystack,
  IN CONST CHAR16 *Needle
  )
{
  if (Haystack == NULL || Needle == NULL) return NULL;
  UINTN needleLen = StrLen(Needle);
  if (needleLen == 0) return (CHAR16 *)Haystack;
  for (CONST CHAR16 *p = Haystack; *p != L'\0'; ++p) {
    if (StrnCmp(p, Needle, needleLen) == 0) return (CHAR16 *)p;
  }
  return NULL;
}

// Extract all port numbers from DevicePath text by scanning for "USB(" tokens.
// Returns a dynamically allocated array of UINT32 (caller FreePool) and sets Count.
// If none found, returns NULL and Count=0
STATIC
UINT32 *
ExtractUsbPortSequenceFromDevicePathText(
  IN CHAR16 *DevPathText,
  OUT UINTN *Count
  )
{
  *Count = 0;
  if (DevPathText == NULL) return NULL;

  // We'll collect into a temporary dynamic array
  UINTN cap = 8;
  UINT32 *arr = AllocateZeroPool(sizeof(UINT32) * cap);
  if (arr == NULL) return NULL;

  CHAR16 *p = DevPathText;
  while (p != NULL && *p != L'\0') {
    // find "USB("
    CHAR16 *found = FindSubstring(p, L"USB(");
    if (found == NULL) break;
    // position after '('
    CHAR16 *q = found + 4;
    // parse first number (port)
    UINT32 port = 0;
    if (!ParseHexNumberAndAdvance(&q, &port)) {
      // skip this token, continue search after found+4
      p = found + 4;
      continue;
    }
    // we parsed port; store it
    if (*Count >= cap) {
      UINT32 *newArr = ReallocatePool(sizeof(UINT32) * cap, sizeof(UINT32) * cap * 2, arr);
      if (newArr == NULL) {
        FreePool(arr);
        *Count = 0;
        return NULL;
      }
      arr = newArr;
      cap *= 2;
    }
    arr[*Count] = port;
    (*Count)++;

    // continue searching after this 'USB(' occurrence
    p = q;
  }

  if (*Count == 0) {
    FreePool(arr);
    return NULL;
  }
  return arr;
}

// -------------------- simple tree structure --------------------

typedef struct _PORT_NODE {
  UINT32 PortNumber;
  EFI_HANDLE DeviceHandle; // if a device is attached exactly at this node
  UINT16 VendorId;
  UINT16 ProductId;
  CHAR16 *DevicePathText; // optional text
  struct _PORT_NODE **Children;
  UINTN ChildCount;
} PORT_NODE;

STATIC
PORT_NODE *
CreatePortNode(UINT32 PortNumber)
{
  PORT_NODE *n = AllocateZeroPool(sizeof(PORT_NODE));
  if (n == NULL) return NULL;
  n->PortNumber = PortNumber;
  n->DeviceHandle = NULL;
  n->VendorId = 0;
  n->ProductId = 0;
  n->DevicePathText = NULL;
  n->Children = NULL;
  n->ChildCount = 0;
  return n;
}

STATIC
VOID
FreePortTree(PORT_NODE *n)
{
  if (n == NULL) return;
  for (UINTN i = 0; i < n->ChildCount; ++i) {
    FreePortTree(n->Children[i]);
  }
  if (n->Children) FreePool(n->Children);
  if (n->DevicePathText) FreePool(n->DevicePathText);
  FreePool(n);
}

STATIC
PORT_NODE *
FindOrCreateChild(PORT_NODE *parent, UINT32 port)
{
  for (UINTN i = 0; i < parent->ChildCount; ++i) {
    if (parent->Children[i]->PortNumber == port) return parent->Children[i];
  }
  // create new child
  PORT_NODE *c = CreatePortNode(port);
  if (c == NULL) return NULL;
  UINTN old = parent->ChildCount;
  PORT_NODE **newArr = ReallocatePool(old * sizeof(PORT_NODE *), (old + 1) * sizeof(PORT_NODE *), parent->Children);
  if (newArr == NULL) {
    FreePortTree(c);
    return NULL;
  }
  parent->Children = newArr;
  parent->Children[old] = c;
  parent->ChildCount = old + 1;
  return c;
}

// Insert a device into the port tree using port sequence
STATIC
BOOLEAN
InsertDeviceByPortSequence(
  PORT_NODE *root,
  UINT32 *Ports,
  UINTN PortCount,
  EFI_HANDLE DeviceHandle,
  UINT16 VendorId,
  UINT16 ProductId,
  CHAR16 *DevicePathText
  )
{
  PORT_NODE *cur = root;
  for (UINTN i = 0; i < PortCount; ++i) {
    cur = FindOrCreateChild(cur, Ports[i]);
    if (cur == NULL) return FALSE;
  }
  // attach device info at leaf
  cur->DeviceHandle = DeviceHandle;
  cur->VendorId = VendorId;
  cur->ProductId = ProductId;
  if (DevicePathText != NULL) {
    cur->DevicePathText = AllocateCopyPool((StrLen(DevicePathText) + 1) * sizeof(CHAR16), DevicePathText);
  }
  return TRUE;
}

STATIC
VOID
PrintPortTreeRecursive(PORT_NODE *node, UINTN depth)
{
  for (UINTN i = 0; i < depth; ++i) Print(L"  ");
  if (node->PortNumber == 0 && node->DeviceHandle == NULL && node->ChildCount == 0) {
    Print(L"<empty root>\n");
    return;
  }
  if (node->PortNumber != 0) {
    Print(L"Port %u", (UINT32)node->PortNumber);
  } else {
    Print(L"Root");
  }
  if (node->DeviceHandle != NULL) {
    Print(L" -> Device Handle=0x%p  VID=0x%04x PID=0x%04x", node->DeviceHandle, node->VendorId, node->ProductId);
  }
  if (node->DevicePathText != NULL) {
    Print(L"  (%s)", node->DevicePathText);
  }
  Print(L"\n");
  for (UINTN i = 0; i < node->ChildCount; ++i) {
    PrintPortTreeRecursive(node->Children[i], depth + 1);
  }
}

// -------------------- main --------------------

EFI_STATUS
EFIAPI
UefiMain (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS Status;
  EFI_HANDLE *PciHandles = NULL;
  UINTN PciCount = 0;

  // locate all PCI handles
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiPciIoProtocolGuid, NULL, &PciCount, &PciHandles);
  if (EFI_ERROR(Status) || PciCount == 0) {
    Print(L"No PCI devices found or error: %r\n", Status);
    return EFI_SUCCESS;
  }

  // collect all USB device handles
  EFI_HANDLE *UsbHandles = NULL;
  UINTN UsbCount = 0;
  Status = gBS->LocateHandleBuffer(ByProtocol, &gEfiUsbIoProtocolGuid, NULL, &UsbCount, &UsbHandles);
  if (EFI_ERROR(Status)) {
    UsbHandles = NULL;
    UsbCount = 0;
  }

  // optional DevicePathToText
  EFI_DEVICE_PATH_TO_TEXT_PROTOCOL *DpToText = NULL;
  gBS->LocateProtocol(&gEfiDevicePathToTextProtocolGuid, NULL, (VOID**)&DpToText);

  Print(L"Scanning %u PCI devices for USB host controllers...\n\n", (UINT32)PciCount);

  for (UINTN i = 0; i < PciCount; ++i) {
    EFI_PCI_IO_PROTOCOL *PciIo = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiPciIoProtocolGuid, (VOID **)&PciIo);
    if (EFI_ERROR(Status) || PciIo == NULL) continue;

    // Read class code/reg at offset 0x08
    UINT32 ClassReg = 0;
    Status = PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, 0x08, 1, &ClassReg);
    if (EFI_ERROR(Status)) continue;

    UINT8 BaseClass = (ClassReg >> 24) & 0xFF;
    UINT8 SubClass  = (ClassReg >> 16) & 0xFF;
    UINT8 ProgIf    = (ClassReg >> 8)  & 0xFF;

    // USB host controllers: BaseClass 0x0C Serial Bus, SubClass 0x03 USB
    if (BaseClass != 0x0C || SubClass != 0x03) continue;

    Print(L"PCI USB Host Controller found (handle=0x%p): Class=0x%02x Sub=0x%02x ProgIf=0x%02x\n",
          PciHandles[i], (UINT32)BaseClass, (UINT32)SubClass, (UINT32)ProgIf);

    // Get PCI device path
    EFI_DEVICE_PATH_PROTOCOL *PciDp = NULL;
    Status = gBS->HandleProtocol(PciHandles[i], &gEfiDevicePathProtocolGuid, (VOID **)&PciDp);
    if (!EFI_ERROR(Status) && PciDp != NULL) {
      if (DpToText != NULL) {
        CHAR16 *Text = DpToText->ConvertDevicePathToText(PciDp, FALSE, FALSE);
        if (Text) {
          Print(L"  PCI DevicePath: %s\n", Text);
          FreePool(Text);
        } else {
          Print(L"  PCI DevicePath: (conversion failed)\n");
        }
      } else {
        Print(L"  PCI DevicePath: (present; no DevicePathToText)\n");
      }
    }

    UINTN PciDpSize = (PciDp != NULL) ? GetDevicePathSizeBytes(PciDp) : 0;

    // Prepare root of port tree for this host
    PORT_NODE *root = CreatePortNode(0); // port 0 = root

    Print(L"  Looking for USB devices under this host (scanning %u USB handles)...\n", (UINT32)UsbCount);
    for (UINTN u = 0; u < UsbCount; ++u) {
      EFI_DEVICE_PATH_PROTOCOL *UsbDp = NULL;
      Status = gBS->HandleProtocol(UsbHandles[u], &gEfiDevicePathProtocolGuid, (VOID **)&UsbDp);
      if (EFI_ERROR(Status) || UsbDp == NULL) continue;

      UINTN UsbDpSize = GetDevicePathSizeBytes(UsbDp);
      if (PciDp != NULL && UsbDpSize >= PciDpSize && DevicePathIsPrefixBinary(PciDp, PciDpSize, UsbDp, UsbDpSize)) {
        // device is child of this PCI host
        // get textual DP (needed to parse USB(...) nodes)
        CHAR16 *UsbDpText = NULL;
        if (DpToText != NULL) {
          CHAR16 *txt = DpToText->ConvertDevicePathToText(UsbDp, FALSE, FALSE);
          if (txt) {
            UsbDpText = txt; // we will FreePool(UsbDpText) later
          }
        }

        // Get Vendor/Product if possible
        UINT16 vid = 0, pid = 0;
        EFI_USB_IO_PROTOCOL *UsbIo = NULL;
        Status = gBS->HandleProtocol(UsbHandles[u], &gEfiUsbIoProtocolGuid, (VOID **)&UsbIo);
        if (!EFI_ERROR(Status) && UsbIo != NULL) {
          EFI_USB_DEVICE_DESCRIPTOR dd;
          ZeroMem(&dd, sizeof(dd));
          if (!EFI_ERROR(UsbIo->UsbGetDeviceDescriptor(UsbIo, &dd))) {
            vid = dd.IdVendor;
            pid = dd.IdProduct;
          }
        }

        // Extract port sequence
        UINTN portCount = 0;
        UINT32 *ports = NULL;
        if (UsbDpText != NULL) {
          ports = ExtractUsbPortSequenceFromDevicePathText(UsbDpText, &portCount);
        }

        // Insert into tree (if we have ports). If no textual DP / no USB() nodes, attach at root with info.
        if (ports != NULL && portCount > 0) {
          InsertDeviceByPortSequence(root, ports, portCount, UsbHandles[u], vid, pid, UsbDpText);
          FreePool(ports);
        } else {
          // no ports found: attach as device on root (special)
          PORT_NODE *leaf = FindOrCreateChild(root, 0); // port 0 = unknown direct child
          leaf->DeviceHandle = UsbHandles[u];
          leaf->VendorId = vid;
          leaf->ProductId = pid;
          if (UsbDpText) {
            leaf->DevicePathText = AllocateCopyPool((StrLen(UsbDpText) + 1) * sizeof(CHAR16), UsbDpText);
            FreePool(UsbDpText);
          }
        }

        if (UsbDpText != NULL) {
          // if we used UsbDpText for insertion and made a copy, free original here
          FreePool(UsbDpText);
        }
      }
    } // usb handles loop

    // Print built tree
    Print(L"\n  USB topology under this host:\n");
    PrintPortTreeRecursive(root, 1);

    // free port tree
    FreePortTree(root);

    Print(L"\n");
  } // pci loop

  if (PciHandles) FreePool(PciHandles);
  if (UsbHandles) FreePool(UsbHandles);

  Print(L"Done. Press any key to exit...\n");
  {
    EFI_INPUT_KEY Key;
    while (gST->ConIn->ReadKeyStroke(gST->ConIn, &Key) == EFI_NOT_READY) {
      // spin
    }
  }

  return EFI_SUCCESS;
}
