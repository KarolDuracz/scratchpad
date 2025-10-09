<h3>1. Maybe I'll start with what works and what has been improved.</h3>
GPT-5 gave me some reasons why it can only do 1x BEEP and then freeze. It's at the bottom of the page. He created a 
helper function called Ac97StopAndResetPcmOut. See the bottom of the page for information on what GPT-5 wrote about this problem and its solution 
(So the first thing that needed to be improved in demo14).
<br /><br />
Does it work? It seems ok, but after several attempts to start it, at some point it stopped playing sound again. But now it doesn't freeze like it did in demo14.

<h3>2. How did I test?</h3>

Same as demo14 but I will put the command into QEMU again

```
qemu-system-x86_64 -L . -bios /share/OVMF.fd -device qemu-xhci,id=xhci -drive if=none,id=usbdisk,file="\\.\PHYSICALDRIVE1",format=raw -cdrom "C:\Users\kdhome\Documents\ImageISO\ubuntu-14.04.6-desktop-amd64.iso" -m 1024 -device usb-storage,drive=usbdisk -audiodev dsound,id=snd0,latency=20000 -device AC97,audiodev=snd0 -machine pcspk-audiodev=snd0 -serial tcp:127.0.0.1:4444,server,nowait
```

And in the same way I put code into the HelloWorld file.

<h3>3. File list</h3>

> [!IMPORTANT]
> THERE IS A DIFFERENCE IN HelloWorld.inf FILE for the demo that loads the WAV file. That's why I placed it in a separate folder. For the demos that play JINGLE BELLS notes, the inf file is the same as for demo 14.

1. HelloWorld - jingle bells demo 1 working 09-10-2025.c --->> demo 1 channel, 1 playing a few notes of jingle bells
2. HelloWorld - jingle bells demo 2 - 10 sec 09-10-2025.c --->> demo 1 channel, 10 sec playing a few notes of jingle bells
3. HelloWorld - 3 channels demo3 - 09-10-2025.c --->> 3 channels demo, 10 sec playing a few notes of jingle bells
4. \a demo that loads WAV\HelloWorld.c --->> demo of loading a WAV file that requires analysis and error fixes
5. demo09102025.wma --->> I recorded the test as in Demo14. It takes 2 minutes.

```
Test 1 channel, 1 repeat -> playing from ~4-5 sec
Test 1 channel, 10 sec -> playing from ~35 sec of recording
Test 3 channel, 10 sec -> playing from ~ 1 min 9 sec of recording
HelloWrold.c -> demo WAV -> playing from  ~ 1 min 37 sec of recording
End -> 1 min 57 sec
```

And the equivalent .efi files. HelloWorld.efi - This is the file that loads the WAV. The rest is the same as the file name.

<h3>4. Demo that tries to load a WAV file - NOT WORKING - and there is a lot of fragmented code, a lot of debugging but I haven't been able to figure it out right now
</h3>
In demo 14, I wrote that I'd do a more in-depth analysis when I do jingle bells demo, but I'll leave that for another day. The last listing, which attempts 
to load a WAV file, will be a good starting point for trying to break it down into smaller pieces and try to understand what's going on.  Next time, I'll need
 to jump to this point. Screenshot shows, that it doesn't detect path to file. This is a flawed implementation. I DIDN'T UPLOAD "jet.wav" FILE HERE. It's 4MB, 
 but if demo doesn't work, it is not necessary for now. <br /><br />

```
  Print(L"Image loaded from handle device = %p (should be fs1:)\n", LoadedImage->DeviceHandle);

  Status = Dir->Open(Dir, &File, L"fs1:\\__bin\\a09-10-2025\\jet.wav", EFI_FILE_MODE_READ, 0); // <<<<<<<<<<<<<<<<<<<<<<<<<< hard code path to file on USB flash drive
  if (EFI_ERROR(Status)) {
    Print(L"OpenFileFromImageDir: Failed to open '%s' in image folder: %r\n", FileName, Status);
    if (Dir != Root) Dir->Close(Dir);
    Root->Close(Root);
    FreePool(dpText);
    return Status;
  }
```

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/tianocore%20EDK2/demo15%20-%20jingle%20bells%20few%20notes%20and%20investigation%20demo%2014%20-DMA%20not%20working%20properly/screenshot%20_%20demo%20with%20WAV%20file.png?raw=true)

<h3>5. For this case why demo 14 can only play 1x BEEP GPT-5 gave me this solution.</h3>

I'll just paste here what he wrote to me

```
The symptom you describe (first run plays fine, subsequent runs are silent even though the app executes) is a classic device-state problem: the AC’97 NABM channel or codec remains in a state after the first run that prevents the controller from accepting a new BDL/run until you explicitly stop/reset the channel and clear status bits. The fix is to explicitly stop and reset the PCM OUT engine before you start each new playback (and also reset/clear after a run finishes), and to probe status registers for diagnostics.

Below I explain why this happens and give a small, safe, well-scoped patch you can drop into your program to make repeated runs robust. The patch does three things:

1. Always stop DMA and perform a channel reset before programming the BDL and starting playback.
2. Clear NATX status bits and POBDBAR/LVI so the controller starts in a clean state.
3. Do the same cleanup after playback finishes so the device is left reusable by subsequent runs.


 Why it happens (short)

* The NABM *Transfer Control* and *status* registers can remain in a state where the controller considers the transfer finished or stuck. Some controllers require a channel reset (Reset bit) to be asserted so internal pointers return to a known state before a new BDL is accepted.
* If you re-run the app without resetting, the controller may ignore the new POBDBAR/LVI or not start because its internal state machine wasn't returned to idle.
* Clearing status bits only once (before the first run) is not always sufficient because the controller's internal BDL pointer or engine state can remain modified by the previous transfer.


 What to add

Add a small helper to stop + reset + wait + clear status. Call it:

* once at the very start of your playback sequence (before writing POBDBAR/LVI),
* and again after you stop DMA at the end (to leave the device clean).

Here is a minimal helper and how to call it. Use the same register offsets you already have.


// Call before programming BDL and before starting DMA, and also after stopping DMA.
STATIC
EFI_STATUS
Ac97StopAndResetPcmOut(
  IN EFI_PCI_IO_PROTOCOL *PciIo
  )
{
  EFI_STATUS Status;
  UINT8 ctrl;
  UINT16 clr;
  UINTN retry;
  UINT8 readVal;

  // 1) Clear Run bit (write 0)
  ctrl = 0x00;
  Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
  if (EFI_ERROR(Status)) {
    Print(L"Ac97: failed to clear RUN: %r\n", Status);
    // continue trying to reset anyway
  } else {
    Print(L"Ac97: POCTRL RUN cleared\n");
  }

  // 2) Assert Reset bit (bit1 = 0x02). This should force the channel back to a known state.
  ctrl = 0x02;
  Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrl);
  if (EFI_ERROR(Status)) {
    Print(L"Ac97: failed to write RESET: %r\n", Status);
  } else {
    // poll until reset clears (hardware usually clears the reset bit when done)
    for (retry = 0; retry < 2000; ++retry) { // up to ~2s polling (with 1ms stall)
      Status = PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &readVal);
      if (EFI_ERROR(Status)) break;
      if ((readVal & 0x02) == 0) break; // reset cleared
      gBS->Stall(1000); // 1 ms
    }
    Status = (EFI_ERROR(Status)) ? Status : EFI_SUCCESS;
    if (EFI_ERROR(Status)) {
      Print(L"Ac97: reset poll read failed: %r\n", Status);
    } else {
      Print(L"Ac97: RESET cleared (poll tried %u times)\n", (UINT32)retry);
    }
  }

  // 3) Clear NATX status word (write the status bits that clear)
  clr = 0x1C; // same value you used; write ones for bits you want clear
  Status = PciIo->Io.Write(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &clr);
  if (!EFI_ERROR(Status)) Print(L"Ac97: cleared NATX status (0x06) := 0x%04x\n", clr);
  else Print(L"Ac97: failed clearing NATX status: %r\n", Status);

  // 4) Zero POBDBAR and LVI so controller doesn't hold stale pointers (optional but safe)
  (VOID)IoWrite32BarIndex(PciIo, 1, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR, 0);
  {
    UINT8 lzero = 0;
    (VOID)PciIo->Io.Write(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POLVI, 1, &lzero);
  }
  Print(L"Ac97: POBDBAR & LVI zeroed\n");

  return EFI_SUCCESS;
}


 Where to call it in your flow

1) At the very start of the section where you “Program controller (POBDBAR, LVI, clear status, start)” — call `Ac97StopAndResetPcmOut(PciIo)` before writing POBDBAR and LVI. This guarantees that stale state from the previous run will be cleared.

2) After playback (after you stop DMA and before unmapping/freeing), call `Ac97StopAndResetPcmOut(PciIo)` again to leave the controller in a clean state.

Example (pseudo-integration):


// before writing POBDBAR/LVI:
Ac97StopAndResetPcmOut(PciIo);

// program bdl, LVI, clear status again if needed, then start
IoWrite32BarIndex(...POBDBAR...);
Write LVI...
clear status...
start DMA...
gBS->Stall(...)

// stop DMA
ctrl = 0;
PciIo->Io.Write(...POCTRL..., 1, &ctrl);

// now cleanup/reset so next run works
Ac97StopAndResetPcmOut(PciIo);


 Extra diagnostics (optional, but helpful)

If you still see problems, add prints that read back the following before/after runs:

* Read back `POCTRL` (byte) and show its value.
* Read back `NABM status` word (0x06) and print it.
* Read back POBDBAR and LVI.

Example reads:


UINT8 ctrlVal;
UINT16 stVal;
UINT32 pob;
PciIo->Io.Read(PciIo, EfiPciIoWidthUint8, 1, NABM_PCM_OUT_BASE_OFF + NABM_POCTRL, 1, &ctrlVal);
PciIo->Io.Read(PciIo, EfiPciIoWidthUint16, 1, NABM_PCM_OUT_BASE_OFF + NABM_STATUS_WORD, 1, &stVal);
PciIo->Pci.Read(PciIo, EfiPciIoWidthUint32, NABM_PCM_OUT_BASE_OFF + NABM_POBDBAR /* if you can access via Pci.Read or Io.Read accordingly */ , 1, &pob);
Print(L"POCTRL=0x%02x STATUS=0x%04x POBDBAR=0x%08x\n", ctrlVal, stVal, pob);


(Use `PciIo->Io.Read` with appropriate widths for POBDBAR if you map BAR1 IO space. In your current code you write POBDBAR via `IoWrite32BarIndex`, so a read with `PciIo->Io.Read` width `Uint32` at same offset will show current value.)

 Why this is safe

* You’re performing only register writes and reads over PCI I/O space and small, bounded polls. No DMA buffers are overwritten.
* Resetting the PCM engine is the standard way to ensure the controller can accept a fresh BDL. Many OS drivers do the same on channel start/stop.

 Final notes / gotchas

* Some codecs / controller combos require also re-writing the sample rate or re-arming the codec if it went to low-power; your current code writes the sample rate at start — keep that.
* If you ever use interrupts to stream longer audio, you must ensure your interrupt handling also clears NATX status bits expected by the controller.
* If you still see one short beep and nothing else, print the BDL entries the controller sees (POBDBAR) and poll NATX status during playback — that will reveal whether the controller consumed only the first BDL entry and then stopped with an error bit set.

```

