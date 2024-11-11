<h2>This is a continuation from the "Custom Windows demo1" folder </h2>

![dump](https://github.com/KarolDuracz/scratchpad/blob/main/bootloader_x86/Custom%20Windows%20demo1%20-%20FTP/188%20-%2011-11-2024%20-%20ftp%20configured%20and%20works.png?raw=true)

<br />
<h2>// GUEST MACHINE CONFIGURATION</h2>
<br /><br />
This is a complex topic and I am still completely newbie when it comes to networking... <br /><br />
After 1 day I find solution. But this is quick guide.<br />
1. First we need to reconfigure VirtualBox Adapter1 from NAT to bridged network card (bridge /  Bridged Adapter) - second option - This set ip addres for guest OS on VirtualBox from 10.0.2.x to 192.168.1.xxx. <br />
2. In this live Windows PE (btw. Windows PE (Preinstallation Environment) does not include the netsh advfirewall commands, as it is a minimal environment focused on troubleshooting, installation, and recovery, and does not have all the features of a full Windows installation, including advanced firewall management) we need to turn off  firewall with the command <b>Wpeutil DisableFirewall</b> https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/wpeutil-command-line-options?view=windows-11 <br />
3. We can't directly change <b>reg add "HKLM\SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters\FirewallPolicy\StandardProfile" /v EnableFirewall /t REG_DWORD /d 0 /f</b> register in this path and set EnableFirwall from 1 to 0 in live mode, like that. This needs reboot system. After <b>Wpeutil Reboot</b> register still has 1. That's why this command is needed <b>Wpeutil DisableFirewall</b> in this live CD mode. --> HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters\FirewallPolicy <br />
4. I created a folder on the host computer in c:\usr_bin\share_file . And in Properties > Sharing (tab) > Advenced Sharing > "share this folder" turn on. (this is necessary?) <br />
5. In "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters" there is information about the TCPIP configuration (in general, information about the system and configuration, mainly in Windows PE, can be found in the registry)
<br /><br />
For guest machine that's all.
<br /><br />
<h2>// HOST machine (in my case Windows 8.1 is host machine)</h2>
<br /><br />
1. I installed for python pyftpdlib

```
pip install pyftpdlib
```

2. Here it is FTP simple server script

```
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

# Subclass DummyAuthorizer to allow blank passwords
class CustomAuthorizer(DummyAuthorizer):
    def validate_authentication(self, username, password, client_address):
        if username == 'anonymous' and not password:
            return True  # Allow anonymous user with blank password
        return super().validate_authentication(username, password, client_address)

def create_ftp_server():

    # Create an instance of the custom authorizer
    authorizer = CustomAuthorizer()

    # Add an anonymous user with read-write permissions
    authorizer.add_anonymous('C:\\usr_bin\\share_files', perm='elradfmw')

    # Instantiate a handler object
    handler = FTPHandler
    handler.authorizer = authorizer

    # Configure passive mode settings
    handler.passive_ports = range(1024, 65535)  # Set a range for passive mode data ports (adjust as needed)

    # You can set an external IP address if the server is behind NAT or running in a VM
    handler.masquerade_address = '192.168.1.102'  # Replace with your server's public/external IP

    # Set up the FTP server with the specified handler and listen on a port
    address = ('0.0.0.0', 21)  # Empty string means the server listens on all interfaces
    server = FTPServer(address, handler)

    # Start the FTP server
    print("FTP server started on port 21...")
    server.serve_forever()

if __name__ == "__main__":
    create_ftp_server()
```

3. Run this server. Change handler.masquerade_address to your host IP address. In my host Windows 8.1 I have 192.168.1.102 right here.

```
python ftp_server.py
```

<br />
<h2>// Guest to host connection </h2>
<br />
First, open new CMD in new window just typing in main windows "start cmd" for netstat, ipconfig etc. Or for FTP connection.

```
X:\Windows\System32\start cmd.exe
```

Go to X:\Windows\temp

```
cmd > cd X:\windows\tmp
```

And then run "ftp" command . When yout start from X:\Windows\System32 etc you can't download files from ftp to this system folder probably.

<br />
Commands from image

```
ftp> // enter to ftp in cmd
ftp> open 192.168.1.102  // open connection to ip host

220 Welcome to the FTP server.
User (host: (none)): anonymous // type anonymous - look at ftp server implementation in validate_authentication function
pass: // empty - press enter only
230 Login successful.

ftp> dir // test ftp command 
```
The FTP server should display a list of files from the folder as in the image above


<br /><br />
Example 
```
ftp <host_ip_address>
Connected to <host_ip_address>
220 Welcome to the FTP server.
User (host: (none)): anonymous
230 Login successful.
ftp> ls
200 EPRT command successful.
150 Opening data connection.
-rw-r--r--    1 0        0            1000 May 10 12:34 example.txt
226 Transfer complete.
ftp> get example.txt
local: example.txt remote: example.txt
200 EPRT command successful.
150 Opening data connection.
226 Transfer complete.
ftp> put newfile.txt
local: newfile.txt remote: newfile.txt
200 EPRT command successful.
150 Opening data connection.
226 Transfer complete.
ftp> bye
```

<br /><br />
This is not complete guide. And many issues is around this. I fighted with this 1 day, so... 
