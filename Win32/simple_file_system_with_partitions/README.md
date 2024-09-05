try to build simple file system with partitions
<br /><br />
In this file is 3 code example. First use zlib. But desn't work. Second which is right now compiled generate error. 
This example creates a secured file  VIRTUAL_DISK_NAME L"C:\\Windows\\Temp\\virtual_disk.bin". And I try to test writing and reading.
The main idea is to create secure file to hashing files into it like text or word files, images etc.
<br /><br />
TODO - to fix
<br />
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Win32/simple_file_system_with_partitions/error_fs_.png)

<br /><br />
Second example - This is commented out at the bottom of the main.cpp file.
<br />
![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/main/Win32/simple_file_system_with_partitions/last_example_fs.png)

btw.
In line 355 there is function <br />
void measurePerformance(const std::function<void()>& operation, const char* operationName) <br />
In many case C++ is helpful, but orignal line looks like that: <br />
void measurePerformance(void (*operation)(void), const char *operationName) <br />
In book "C programming language second edition" there is example about extended declaration<br />
https://github.com/gerrard00/the-c-programming-language/blob/master/test-dcl.sh <br >
https://github.com/Heatwave/The-C-Programming-Language-2nd-Edition/tree/master <br />
I like this book and this repositories with practical example and tiny demos <br />
