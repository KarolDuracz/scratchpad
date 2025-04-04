# scratchpad
$${\color{red}}$$	
		${{\color{red}\Huge{\textsf{     for\   educational\   reason \  only\   [CLOSED]}}}}\$


Update 10.02.2025 : For all demos and notes from 2025 at the end I will put a more detailed summary. Mostly for myself to sumarize all topics. But maybe someone find here something for on his own path of learning programming, like everything on github. For now I'm creating a timeline for myself. So far there are many errors. But for the purposes of a quick reminder of the topic by days, weeks it is fine for me. I DO NOT RECOMMEND RUNNING THESE CODES WITHOUT VERIFICATION. THIS IS JUST A QUICK INTRO TO CHECK IF IT WORKS. (...) I think that's enough for now. I've checked some UEFI protocols on my hardware. Few WDM demos. This shows that it works. Next upload in mid-2025. From the plans, it is an analysis of drivers for Intel HD 3000 and Nvidia GT540M. But a general analysis of basic functions. I don't want to play too much with it, it's risky. (...) I had to look at the datasheet Intel® Core™ Ultra 200V Series Processors Datasheet, Volume 1 of 2  to see how big gap I had in my hardware. To make sure that what I do here makes sense. But it's not bad, there is backward compatibility.
 <br /><br />
Update 18.03.2025 : OK. Goal for 2025 is building a Virtual Machine as a warm-up because I haven't done anything specific for a long time https://github.com/KarolDuracz/scratchpad/tree/main/bootloader_x86/tianocore%20EDK2 . But what exactly, apart from making a VM like Qemu, VirtualBox? There is something else. Apart from that, can I connect LLM between hardware and OS? I am also interested in whether it is possible to add JIT which will copy instructions from the process memory, convert instructions that take a long time to execute to AVX, OpenCL, CUDA etc. That is, on the fly it will take pieces of instructions from the process memory and execute this code only on AVX for example to speed up and return result and then back to execute original process code. One of the applications that I want to test in this way is Age Of Empires II Microsoft 1999. I am interested in whether it is possible to accelerate part of the instructions in this way. The second issue is code debugging. From the perspective of the VM, you can "see" the memory of the entire system and maybe I can look deep into the application. Also CPU instructions. And if I add LLM to this and how it understands binary code, it can decode the machine code and understand it. So maybe I can create a better debugger for the entire system and applications running on it. But this is more of a long-term goal. For now, I need to make an equivalent of Qemu / Virtual Box as a warm-up. There are a few other things like creating virtual devices. When I first thought about it, I wanted to have better debugging tools. But I don't know if it's possible to achieve those goals. But I can try... (...) For now, I don't know if I'll make the next upload in the middle of the year or only at the end to summarize all. It would be better to calmly do everything now and upload the finished at the end of the year. 
 <br /><br />
 Update 04.04.2025 : It's gonna be a long and lonely ride...... :sweat_smile: When a few months ago I was analyzing possible scenarios to improve my programming skills. I chose to learn UEFI/BIOS and build my own virtual machine. And what can I say today, a few months later? That this is just the "tip of the iceberg" :flushed: :disappointed_relieved:. Because after a preliminary analysis of how the modules for EmulatorPkg and OMVF are built, their "entry points", what the boot flow looks like, etc. After a preliminary analysis I know one thing - It's gonna be a lonely long ride. Because after I build a VM, i.e. to understanding at least the basics of QEMU, VirtualBox, how OVMF.fd or EmulatorPkg works. After that, there is e.g. UEFI for Raspberry PI 3/4/(5?) -> mobile devices (android)?. And there is https://github.com/tianocore/edk2-platforms/tree/master which is Intel’s Coreboot + EDK2 Payload or EDK2 Platforms + SiliconPkg from hardware vendors. It means, FOR REAL HARDWARE packages / motherboards. And only here will the truly difficult problems probably appear, because topics such as "must integrate Intel FSP binary properly (called during PEI phase)", "Without FSP, your firmware won’t initialize memory, so it won't boot.", "Real-world firmware often includes ME firmware, EC code, and descriptor regions (not always open source)", VBT (Video BIOS Table) - Configuration data needed by the graphics driver and so on etc. But.... I think it's worth going through at least some of this material. Because it really makes me do gymnastics with coding, and I really have to learn it. And above all, look for SCALABILITY in action to make progress. So it's a good exercise in general. (...) That's why I wrote earlier that it would be better to make the next update at the end of '25. :no_mouth: (...) I could upload more demos, e.g. ACPI, minimalist EDK2 build.py script, edksetup.bat etc. because I have something to start "from scratch" with minimal ones, write Uefi.h from scratch and so on. So Print from scratch, GOP from scratch etc. BUT THAT IS MISSING THE PURPOSE. Because such pieces WILL NOT HELP ME OR ANYONE WHO WILL READ THIS IN THE FUTURE.
<hr>
update : 05.09.2024 : This "scratchpad" repo is theoretically closed. I won't be posting anything more here. These are just a few topics that interest me. Probably end of 2025 I'll be back to posting if there's something interesting finally worth sharing and developing. TIME TO GO TO NEXT LEVEL ...in programming skills also. 
<br /><br />
<b>update : 07.09.2024 : About "fix typo on README.md and inside code comments" - I apologize for my low quality and many of a typo mistakes. After some time I feel like I could have put more effort into it and spell checked it before posting the code, but I won't be correcting every comment anymore because it's unnecessary. It's about the code, and the demos. I think the comment captures the general gist or context of what is going on here. It is sufficiently readable and understandable. Besides, it is a "scratchpad" for me. Not a public repo. But like I said, I feel like I could have put more care and effort into it.</b>
<br /><br />
`# IMPORTANT #` update : 11.09.2024 : I do not intend to copy the work of others. Steal someone's work. I DO NOT DO THIS HERE. Links to repositories and materials which I put here is for build memory map for my learning path about programming. For me this is very useful stuff. This is for me only to build knowledge and make progress.
<br /><br />
update : 14.09.2024 : still open, but I stop adding new topics here. This is only for educational reason.
<br /><br />
update : 18.09.2024 : added last few notes to repos for myself in README. At the final I added example from  ammo.js webgl_demo_test_ray to machine learning directory. This is base for further learning. 
<br /><br />
update : 24.09.2024 : TODO in 2025 to get more serious tasks with ML. (1) language translator similar to translate.google (2) minimalist project similar to whisper to encode speech from wideo/music to text - more advanced is translator i.e. en -> pl, en -> ua, en -> ru, ru -> pl, ua -> pl. I need some mix between deutch, english, russian, ukrainian (3) run GPT-2 124M (https://github.com/karpathy/llm.c/discussions/481), 774M (https://github.com/karpathy/llm.c/discussions/580) and 1.6B (https://github.com/karpathy/llm.c/discussions/677) model on 8 x A100 and get some expiriences from that. First I need run 124M model. This is a minimum target. For this purpose I don't need 8xA100, but, the case is to learn MPI and stuff like that to learn more advanced parallel computing. But this is not important right now. I need start somewhere. Model with 124M parameters is the first target to achieve. A lot of ready stuff is here https://www.tensorflow.org/tutorials/audio/simple_audio?hl=pl and https://pytorch.org/tutorials/ but... you have to take it in your hands and process it... I don't know if it will be possible to achieve it, till the end of 2025, but keep in mind. I need learn some basics, because I want. <b>Because after so many years of doing what others want from me, I want to finally start doing what gives me pleasure and satisfaction. That's it. </b>
<br /><br />
update : 25.09.2024 : note in Embedding_RPI4_ . Keep in mind only. Maybe one day I'll come back to this, I'd like to learn more about it.
<br /><br />
update : 28.09.2024 : When I added Win32/web_renderer/ repo I think, At the end of 2025 this all things it need to be DONE! Everything needs to be done step by step and probably thrown into a small system emulator.
Some time ago, looking at these types of projects, I thought it was a waste of time, but I need to put it together, because it's been cluttering my head for ~10 years. I just need exercises like this to get it over with. Then I can do something else with my head free from all this...  So, probably in 2025 I'll work it out and at the end I'll upload a new repo where everything will be accumulated. Maybe some small documentation will be created for the future. (...) Or I'll just implement it for Linux and Windows. I'll think about it, but I have to put it together. Do these exercises.
<br /><br />
update : 4.11.2024 : I uploaded 3 another demos to win32 repo... But when I look at it, it doesn't make sense for me to upload it any more. This scratchpad was supposed to be used to start doing something towards learning certain topics that interest me. But when I look at it now 1) it has terribly low quality and many errors and language typos 2) It wastes a lot of space on github in such a form as those .gif animations that I uploaded and a lot of junk that I treated as a notebook - better for this case create own website or just a folder on a disk on my computer and when I refine certain things to the stage that it makes sense and you don't have to compile it as a project in MSVC but via cl and you can easily explain the compilation and operation 3) "Scratchpad" was supposed to be used to simply collect a few things in one place to get back to learning programming and topics around IT. And this task was fulfilled! (...) I'm looking at this scratchpad/Win32/Multi Windows Manager/ repo now and there are a lot of similar things like hooking window procedures, MouseProc, KeyboardProc and using DefWindowProc, SetWindowsHookEx or like the part from main.cpp where it's SetWindowLongPtr(hChild, GWLP_WNDPROC, (LONG_PTR)ProcessDialogProc2); - it just needs more explanation of how it works from the low level than just copy and paste to compile and run simple demo. Because if you start a new topic, there are a lot of questions, new problems, coding rules, common mistakes, etc. and that is just as important in educational path. And recording this and the conclusions "what did I learn from this". <br /><br />
So. <b>Ok, I stop here!</b> Today I understood that it is better to create a folder on the disk on a local computer, finish demos and then possibly upload everything when it will have educational value for me that I will be able to come back to after years and it will make me actually make progress thanks to this "work".
<br /><br />
Update 11-12-2024 - I added WinPE, tianocore EDK2 and ASL/AML Intel quick intro. At first I threw in some pieces of code to do something as a warm-up. At the end of 2024 I think it's time to change the approach to uploading ready, finished work. It was mainly planning what I have to do now, to remind myself of the basics. There has to be some purpose to it. It can't be learning for the sake of learning, but I have to push myself because this is not trivial. <b>Typical, of what I was used to so far.</b>. Such things are not done in public, in fact I despise such challenges if someone does it in public, but in a way it can help. But only up to a point. Because some things are better done quietly. After a while, this type of "challenge", which was originally intended to serve as a journal, begins to take time, and instead of spending time studying and improving your skills in what you do, you focus more on what you publish. For example. Will learning to play the guitar and posting it on YT make you better guitar player like Jimi Hendrix, Slash, etc.? Nope. No, it doesn't. It's just practice and you either have the talent for it or you don't. This is just an example to have a point of reference.(...)  <b>But since I already have this repo on github, I need to "summarize it somehow".</b> Because, willingly or unwillingly, I started publishing certain things and thoughts.
<br /><br />
Update 23-12-2024 - OK, last word in '24. Something inside tells me that I would rather focus on it now, as long as there is an opportunity for free learning and development myself. I just prefer to focus on it for a year and in the end I'll find out whether I know something and can do it and it makes sense or not. And when I look at, for example, where I can then train Machine Learning, I chose well to create a "playground" on what I have already done (3d board folder). I will grasp the basics faster knowing what I want from the application. It's just about training and pushing myself, and building a Virtual Machine seems like a good challenge for this because it touches on general issues as a whole when it comes to computers (from the software development side, not from the hardware and chip building side). And in a year we will find out whether it is good or nonsense. It is that simple...
<br /><br />
That's it for 2024 !!!
<br />
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
<br /><br />

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

![ntqueryinfo_example2.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/ntqueryinfo_example2.png)

update 23-07-2023 - 
<b>NtQueryInformationProcess_run_example</b>
<br />
Linie 145-147 - odczytanie linii poleceń. To jest przydatne gdy aplikacja jest wywoływana z argumentami, a tak jest np przy otwarciu pliku w notatniku (nie pustego notatnika, tylko pliku .txt). Wtedy w lini poleceń jest ścieżka do pliku. Tak samo proces msedge.exe, czyli przeglądarka internetowa, w lini poleceń jest sporo informacji do wyciągnięcia. <br />

```
ReadProcessMemory(h, rupp.CommandLine.Buffer, cmd, rupp.CommandLine.Length - 2, &rb4);
printf("%d %ws\n", rb4, cmd);
```

<hr>

![EnumWindows_img.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/EnumWindows_img-a.png)

Jeśli chciałbym znaleźć aktualnie otwartą kartę w przeglądarce, aktualnie aktywne okno i to co widnieje w "title" to jak to zrobić? Okazuje się że EnumWindows w tej formie wyciągnie te informacje, ale potrzeba moim zdaniem jest jeszcze kolejna funkcja PrintProcessNameAndID która zwraca ściżkę do procesu (ścieżkę uruchomionego programu np. msedge.exe) oraz PID. Dzięki temu można w dalszym kroku powiązać ze sobą te dwa wyniki zwracane przez funkcje i filtrować (wyszukiwać) konkretnego PID, w tym przypadku dla procesu "msedge.exe", jeśi chcę znać aktualną nawzę otwartego okna. To jest tylko mały przykład od czego zacząć.

![tasklist verbose.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/taskslit%20verbose-a.png)

Gdyby użyć polecenia CMD <b>tasklist /v</b> to nie dostaniemy wystarczajaco informacji o otwartym oknie. Niektóre otwarte strony nawet nie zwrócą tutaj żadnyuch informacji tylko zobaczymy N/A przy wywołaniu tego polecenia. Dlatego trzeba użyć funkcji z winapi żeby dostać się chociaż do szczątkowych informacji o tym jaka jest aktualna nazwa okna / wyswietlana strona w przeglądarce.

<hr>

![tasklist verbose.png](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/another_way_to_track_windows.png)

Może to jest najlepsza opcja żeby śledzić otwarte okna w systemie Windows. Jest najbardziej oczywista. Oprócz nazwy okna, PID procesu w drugiej lini jest nazwa procesu, żeby miec pewność który dokładnie proces był w danym momencie był zapisywany w logach / które okno było aktualnie na pierwszym planie wyświetlane. Czyli dostajemy w ten sposób nazwy otwartych stron (nie adresy, ale nazwy z title) oraz nazwę i PID procesu.

```
#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <iostream>
#include <windows.h>
#include <stdio.h>
#include <psapi.h>
#include <tchar.h>

#define BUF_SIZ 500

FILE* stream_log;

void save_to_file(SYSTEMTIME lt, char *retbuf, int counter)
{
	// logs 
	char log_buf[140] = { 0 };
	char log_tmp_buf[100] = { 0 };
	strcat(log_buf, "c:\\__bin\\detect_ext_logs\\v2-");

	memset(log_tmp_buf, 0, 100);
	_itoa(lt.wYear, log_tmp_buf, 10);
	strcat(log_buf, log_tmp_buf);
	strcat(log_buf, "-");

	memset(log_tmp_buf, 0, 100);
	_itoa(lt.wMonth, log_tmp_buf, 10);
	strcat(log_buf, log_tmp_buf);
	strcat(log_buf, "-");

	memset(log_tmp_buf, 0, 100);
	_itoa(lt.wDay, log_tmp_buf, 10);
	strcat(log_buf, log_tmp_buf);


	memset(log_tmp_buf, 0, 100);
	strcat(log_buf, ".txt");

	//fopen_s(&stream, log_buf, "a");

	printf("%s \n", log_buf);

	fopen_s(&stream_log, log_buf, "a");
	fprintf(stream_log, "%d - %02d | %02d:%02d:%02d ", counter, lt.wDayOfWeek, lt.wDay, lt.wMonth, lt.wYear);
	fprintf(stream_log, "%02d:%02d:%02d", lt.wHour, lt.wMinute, lt.wSecond);
	//fprintf(stream_log, "%s\n", " --logs only.");
	fprintf(stream_log, " %s\n", retbuf);
	fclose(stream_log);
}

int main()
{

	//HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, 9200);

	volatile int i = 0;

	// temporary buffer for first get window name
	char* buffer = new char[BUF_SIZ];

	// the same size - 500
	char* retbuf = new char[BUF_SIZ];

	char buf_name[BUF_SIZ];

	SYSTEMTIME st;

	while (1) {

		HWND hWnd = GetForegroundWindow();

		GetWindowTextW(hWnd, (LPWSTR)buffer, BUF_SIZ);

		int len = GetWindowTextLengthA(hWnd);

		//printf("size of buffer %d %d\n", sizeof(buffer), len);
		
		int j;
		int jj = 0;
		for (j = 0; j < len*2; j++) {
			if (buffer[j] == 0) continue;
			//printf("%c", buffer[j]);
			retbuf[jj] = buffer[j];
			jj++;
		}
		retbuf[jj++] = '\0';

		GetLocalTime(&st);

		DWORD pid = 0;

		GetWindowThreadProcessId(hWnd, &pid);

		printf("%d [%d] - %d:%d:%d - %s\n", i, pid, st.wHour, st.wMinute, st.wSecond,  retbuf);

		HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, pid);

		DWORD dump_pid_name = GetProcessImageFileNameA(hProcess, (LPSTR)buf_name, BUF_SIZ);

		printf("%d %s \n", dump_pid_name, buf_name);

		//save_to_file(st, retbuf, i);

		//printf("%d %ws \n", i, buffer);

		// inc logs counter
		i++;

		Sleep(1000);
	}

	return 0;
}
```
