# scratchpad

NtQueryInformationProcess_run_example - dump.png
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/dump.png)

<b>64 bit - dla 32 bitowego procesu offsety są inne i nie wczyta adresu PEB a co z tym idzie reszty odczytywanej przez ReadProcessMemory.</b>

<i>Dodaj ten kod po lini 85, czyli po printf("%zu \n", sizeof(PVOID));</i><br>
Te struktury i pola bitowe jak IMAGE_DOS_SIGNATURE są w winnt.h
```
IMAGE_DOS_HEADER *idh = (IMAGE_DOS_HEADER*)hdr;			
printf("e_magic: %p %p %d %d %p\n", idh, image, IMAGE_DOS_SIGNATURE, idh->e_magic, idh->e_lfanew);

IMAGE_NT_HEADERS32 *inh32 = (IMAGE_NT_HEADERS32*)((BYTE*)hdr + idh->e_lfanew);
printf("%d %d \n", inh32->Signature, IMAGE_NT_SIGNATURE);
```
Zwróci coś takiego:

e_magic: 000000B6B3A0E0F0 00007FF6B9400000 23117 23117 00000000000000F8<br />
17744 17744
<br/>
Czyli pobrane pola są takie same jak IMAGE_DOS_SIGNATURE i IMAGE_NT_SIGNATURE. W ten sposób można dostawać się do pól tych struktur.
http://pinvoke.net/default.aspx/Structures.IMAGE_DOS_HEADER
Ten offset 0xf8 z idh->e_lfanew jest ten sam który obliczyłem "ręcznie" tylko mi wyszło 247.
<br /><br />
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-image_optional_header32
```
IMAGE_NT_HEADERS32 *inh32 = (IMAGE_NT_HEADERS32*)((BYTE*)hdr + idh->e_lfanew);
printf("%d %d \n", inh32->Signature, IMAGE_NT_SIGNATURE);
printf("%p %d %p %d\n", inh32->OptionalHeader.AddressOfEntryPoint,
inh32->OptionalHeader.SizeOfImage, inh32->OptionalHeader.ImageBase, inh32->OptionalHeader.SizeOfCode);
```
0000000000011276 159744 0000000000007FF7 37376
<hr>
update 23-07-2023 - 
<b>NtQueryInformationProcess_run_example</b>
<br />
Linie 145-147 - odczytanie linii poleceń. To jest przydatne gdy aplikacja jest wywoływana z argumentami, a tak jest np przy otwarciu pliku w notatniku (nie pustego notatnika, tylko pliku .txt). Wtedy w lini poleceń jest ścieżka do pliku. Tak samo proces msedge.exe, czyli przeglądarka internetowa, w lini poleceń jest sporo informacji do wyciągnięcia. <br />

```
ReadProcessMemory(h, rupp.CommandLine.Buffer, cmd, rupp.CommandLine.Length - 2, &rb4);
printf("%d %ws\n", rb4, cmd);
```
