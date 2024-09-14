Simple CUDA test on NVIDIA Tesla T4 for 1-2 hours but it's better than nothing for the purposes of learning the basics of CUDA. 
On Google Colab servers you can check profiling and other things, so it's pretty good. <br /> <br />
https://github.com/karpathy/llm.c/issues/562 <br />
https://colab.research.google.com/drive/1pFCqlGkfJiIr1HLTgaMh4i5OtiY7P96P?usp=sharing<br />
<br /><br />
%cd /content/drive/MyDrive/Cuda

!pwd
!ls

!which nvcc

!ls -l /dev/nv*
!git clone https://github.com/karpathy/llm.c.git

%cd llm.c

!ls

!git pull

!pip install -r requirements.txt

!python dev/data/tinyshakespeare.py

!python train_gpt2.py

!make train_gpt2fp32cu

!./train_gpt2fp32cu
<br /><br />
<b>to write new file / update file</b> 
%%writefile --> and after that type code in next lines

https://stackoverflow.com/questions/64297543/writefile-magic-command-in-regular-python
