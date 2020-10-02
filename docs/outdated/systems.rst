
Systems
=========

Systems where Opticks has been Installed
------------------------------------------

macOS 10.13.4 (17E199) High Sierra, Xcode 9.2  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* macOS 10.13.4 (17E199) High Sierra 
* Xcode 9.2 (actually on 9.3 but xcode-select back to 9.2) as required by nvcc (the CUDA compiler)
* NVIDIA GPU Driver Version: 387.10.10.10.30.103  (aka Web Driver)
* NVIDIA CUDA Driver : 387.178
* NVIDIA CUDA 9.1
* NVIDUA OptiX 5.0.1


macOS 10.9.4 Mavericks : Xcode/clang toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Primary development platfom : Mavericks 10.9.4 
* NVIDIA Geforce GT 750M (mobile GPU) 

Linux : GCC toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~

* DELL Precision Workstation, running Ubuntu 
* DELL Precision Workstation, running CentOS 7
* NVIDIA Quadro M5000 

Windows : Microsoft Visual Studio 2015, Community edition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Ported to Windows 7 SP1 machine 
* non-CUDA capable GPU

Opticks installation uses the bash shell. 
The Windows bash shell that comes with 
the git-for-windows project was used for this purpose

* https://github.com/git-for-windows
 
Despite lack of an CUDA capable GPU, the OpenGL Opticks
visualization was found to operate successfully.


