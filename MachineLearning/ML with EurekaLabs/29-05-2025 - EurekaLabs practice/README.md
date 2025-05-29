I'll take a break from the ML topic until September '25 ~. Something around that.
<br /><br />
I don't know if this makes sense, but I'm wondering how to "bite" the topic of SELF ATTENTION and POSITIONAL ENCODING. Maybe this demo with some fixes will be useful. I looking for a good solution that will analyze such repositories as EDK2, linux etc. It's helps in some way, becasue goes into the root folder with files, in this case .py. And looks for certain tokens after regex and creates links to them recursively throughout the project. (...) At the moment it works fine on python code. There is also a .c version but it doesn't work very well. Image from repo analysis > https://github.com/karpathy/nanoGPT . To be precise, the lines ~52 https://github.com/karpathy/nanoGPT/blob/master/model.py#L52

![dump](https://raw.githubusercontent.com/KarolDuracz/scratchpad/refs/heads/main/MachineLearning/ML%20with%20EurekaLabs/29-05-2025%20-%20EurekaLabs%20practice/screen%2029-05-2025%20ml%20eureka.png)

<h2>How to use</h2>
Base command. 3 arguments after python call.

```
python xref.py path/to/repo output_html
```

Ok, files xref3.py and xref5.py do the same thing, the difference is that xref5.py shows all the lines in the current file and I think history panel on the right side it has an some fixes. This is a sample code to paste in CMD for c_xref.py, i.e. to ANALYZE .C FILES IN REPO

```
cd C:\Users\kdhome\Documents\progs\edk2_win81\analyzer edk2
python c_xref4.py ^
  "C:\Users\kdhome\Documents\progs\EurekaLabs\08-05-2025 - analyze linux repo\linux-master\linux-master\kernel" ^
  "c_xref4_html"
```

For nanGPT, in my case. This is root folder. ( https://github.com/karpathy/nanoGPT/tree/master ) 

```
python xref5.py C:\Windows\Temp\nanoGPT-master\nanoGPT-master nanoGPT_html
```
