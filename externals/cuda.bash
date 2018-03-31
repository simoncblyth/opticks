# === func-gen- : cuda/cuda fgp externals/cuda.bash fgn cuda fgh cuda
cuda-src(){      echo externals/cuda.bash ; }
cuda-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cuda-src)} ; }
cuda-vi(){       vi $(cuda-source) ; }
cuda-env(){      olocal- ; cuda-path ; }
cuda-usage(){ cat << EOU

CUDA
======

See Also
---------

* cudatoolkit-


After 9.1 install on epsilon
------------------------------

In *Sys Prefs > CUDA*::

    [Install CUDA Update] greyed out
    No newer CUDA Driver available
    CUDA Driver Version: 387.128   [Update Required] 
    GPU Driver Version: 355.11.10.10.30.120

::

    epsilon:~ blyth$ cuda-osx-kextstat
      152    0 0xffffff7f80d08000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) 4329B052-6C8A-3900-8E83-744487AEDEF1 <4 1>

Dont find any new kext?::

    epsilon:samples blyth$ ll -t /System/Library/Extensions/



Samples Build OK with switch back to xcode-92 : NB dont have the corresponding headers in /usr/include
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    xcode-;xcode-92;xcode-check

    cuda-samples-make    # after dir setup and mving samples from HOME to there



About CUDA driver 387.128
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Searching for the CUDA driver : http://www.nvidia.com/object/macosx-cuda-387.128-driver.html

::

    Version: 387.128
    Release Date: 2018.01.25
    Operating System: Mac OS
    Language: English (U.S.)
    File Size: 39.7 MB

    New Release 387.128

    CUDA driver update to support CUDA Toolkit 9.1, macOS 10.13.3 
    and NVIDIA display driver 378.10.10.10.25.156
    macOS CUDA driver version format change
    The macOS CUDA driver version now uses the format xxx.xx compare to x.x.x
    to be consistent with our Linux and Windows driver version naming convention.

    Recommended CUDA version(s):
          CUDA 9.1
    Supported macOS
         10.13.x
    An alternative method to download the latest CUDA driver is within macOS
    environment.  Access the latest driver through System Preferences > Other >
    CUDA.  Click 'Install CUDA Update'



QUADRO & GEFORCE MACOS DRIVER RELEASE 387.10.10.10.25.156
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.nvidia.com/object/mac-driver-archive.html
* http://www.nvidia.com/download/driverResults.aspx/130460/en-us

::

    Version:	387.10.10.10.25.156
    Release Date:	2018.1.24
    Operating System:	macOS High Sierra 10.13.3
    CUDA Toolkit:	9.1
    Language:	English (US)
    File Size:	60.95 MB


*CUDA Application Support:*

In order to run macOS Applications that leverage the CUDA architecture of
certain NVIDIA graphics cards, users will need to download and install the
driver for Mac located here.

*Installation Note:*

Because of improvements in macOS security, the Security & Privacy Preferences
may open during the installation process. If it does, click “Allow” in order
for the NVIDIA Graphics Driver to load, then return to the Installer and click
“Restart”.

*New in Release 387.10.10.10.25.156:*

* Graphics driver updated for macOS High Sierra 10.13.3 (17D47)
* Contains performance improvements and bug fixes for a wide range of applications.
* Includes NVIDIA Driver Manager preference pane.
* Includes BETA support for iMac and MacBook Pro systems with NVIDIA graphics

*Release Notes Archive:*

This driver update is for::

    Mac Pro 5,1 (2010), 
    Mac Pro 4,1 (2009) and 
    Mac Pro 3,1 (2008) users.

BETA support is for::

    iMac 14,2 / 14,3 (2013), 
    iMac 13,1 / 13,2 (2012) and
    MacBook Pro 11,3 (2013),       <<<<< 
    MacBook Pro 10,1 (2012), and 
    MacBook Pro  9,1 (2012) users.

Installing the Driver
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:~ blyth$ l ~/Downloads/WebDriver-387.10.10.10.25.158.pkg
    -rw-r--r--@ 1 blyth  staff  63913283 Mar 31 18:51 /Users/blyth/Downloads/WebDriver-387.10.10.10.25.158.pkg

    ## thus one said 10.13.4 not compatible 
       
    epsilon:~ blyth$ l ~/Downloads/WebDriver-387.10.10.10.30.103.pkg 
    -rw-r--r--@ 1 blyth  staff  63913281 Mar 31 18:55 /Users/blyth/Downloads/WebDriver-387.10.10.10.30.103.pkg


Thank you for purchasing an NVIDIA® graphics card for Mac. 

This software package contains:

* A driver designed for your NVIDIA graphics card.
* A preference pane in System Preferences that includes options to:
  
  * Switch between the NVIDIA Web Graphics Driver or Default macOS Graphics Driver.
  * Enable Error Correcting Codes (ECC) on supported graphics cards.
  * Automatically check for NVIDIA software updates for your graphics card.
  * An optional menu bar item that provides easy access to the most common preferences.

If, for any reason, you are unable to boot your system to the desktop and wish
to restore your original drivers, you can do so by clearing your Mac’s NVRAM.
To clear NVRAM, power on your Mac and hold down Command (⌘), Option, P, and R
simultaneously before the gray screen appears. Keep the keys held down until
you hear the startup chime for the second time. The system will now boot with
the original drivers restored.

This will take : 196.7 MB 



OptiX 5.0.1 Release Notes Recommends CUDA 9.0
--------------------------------------------------

* https://developer.nvidia.com/cuda-90-download-archive

CUDA 9.0 Mac
~~~~~~~~~~~~~~

* http://developer.download.nvidia.com/compute/cuda/9.0/Prod/docs/sidebar/CUDA_Installation_Guide_Mac.pdf

 
To use CUDA on your system, you need to have:

* a CUDA-capable GPU
* Mac OS X 10.12
* the Clang compiler and toolchain installed using Xcode

::

   Xcode 8.3.3 
   Apple LLVM 8.1.0   
   macOS 10.12      (Sierra)


CUDA 9.1 Mac
~~~~~~~~~~~~~~

* http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

Driver support for Mac OS 10.13.3 is added in this release.
Xcode 9.2 is now supported as a host compiler on Mac OS.

* https://developer.nvidia.com/cuda-downloads?target_os=MacOSX&target_arch=x86_64&target_version=1013&target_type=dmgnetwork
* https://developer.download.nvidia.com/compute/cuda/9.1/Prod/docs/sidebar/CUDA_Installation_Guide_Mac.pdf

1.1. System Requirements

To use CUDA on your system, you need to have:

* a CUDA-capable GPU
* Mac OS X 10.13
* the Clang compiler and toolchain installed using Xcode
* the NVIDIA CUDA Toolkit (available from the CUDA Download page)

::

    Xcode 9.2
    Apple LLVM  9.0.0
    macOS 10.13.2   (High Sierra)

A supported version of Xcode must be installed on your system. The list of supported
Xcode versions can be found in the System Requirements section. The latest version of
Xcode can be installed from the Mac App Store.
Older versions of Xcode can be downloaded from the Apple Developer Download Page.
Once downloaded, the Xcode.app folder should be copied to a version-specific folder
within /Applications. For example, Xcode 6.2 could be copied to /Applications/
Xcode_6.2.app.
Once an older version of Xcode is installed, it can be selected for use by running the
following command, replacing <Xcode_install_dir> with the path that you copied
that version of Xcode to::

    sudo xcode-select -s /Applications/<Xcode_install_dir>/Contents/Developer


Installer::

  CUDA Driver  (kext)
  CUDA Toolkit
  CUDA Samples



CUDA CMake
------------

* https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake/


Avoid acos
------------

* https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/


Capture stdout logging
-----------------------

* https://stackoverflow.com/questions/21238303/redirecting-cuda-printf-to-a-c-stream


OSX CUDA Driver
-----------------

August 2016
~~~~~~~~~~~~~

CUDA 7.5.30 Driver update is available

CUDA Driver Version 7.0.29
GPU Driver Version 8.26.26 310.40.45f01


SDU
-----

* http://www.anandtech.com/show/8729/nvidia-launches-tesla-k80-gk210-gpu

::

    [simon@GPU ~]$ cp -R /usr/local/cuda-8.0/samples/  cuda-8.0-samples
    [simon@GPU cuda-8.0-samples]$ which nvcc
    /usr/local/cuda-8.0/bin/nvcc
    [simon@GPU cuda-8.0-samples]$ make

    ...
    [@] mkdir -p ../../bin/x86_64/linux/release
    [@] cp simpleGL ../../bin/x86_64/linux/release
    make[1]: Leaving directory `/home/simon/cuda-8.0-samples/2_Graphics/simpleGL'
    >>> WARNING - libGL.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<
    >>> WARNING - libGLU.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<
    >>> WARNING - libX11.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<


    simon@GPU cuda-8.0-samples]$ bin/x86_64/linux/release/deviceQuery
    bin/x86_64/linux/release/deviceQuery Starting...

     CUDA Device Query (Runtime API) version (CUDART static linking)

    Detected 4 CUDA Capable device(s)

    Device 0: "Tesla K80"
      CUDA Driver Version / Runtime Version          8.0 / 8.0
      CUDA Capability Major/Minor version number:    3.7
      Total amount of global memory:                 11440 MBytes (11995578368 bytes)
      (13) Multiprocessors, (192) CUDA Cores/MP:     2496 CUDA Cores
      GPU Max Clock rate:                            824 MHz (0.82 GHz)
      Memory Clock rate:                             2505 Mhz
      Memory Bus Width:                              384-bit
      L2 Cache Size:                                 1572864 bytes

    ...




DOCS
-------

* http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf

CUDA Linux
-------------

* http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Native Linux Distribution Support 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::
    
    Disto       Kernel  GCC     GLIBC 

    CUDA 8.0, 7.5 same

    RHEL 7.x    3.10    4.8.2   2.17    
    RHEL 6.x    2.6.32  4.4.7   2.12


EPEL : Extra Packages for Enterprise Linux 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://fedoraproject.org/wiki/EPEL





CUDA Release History
----------------------

* https://developer.nvidia.com/cuda-toolkit-archive

::

    9.1.199 : RN-06722-001 _v9.1 | March 2018 

    CUDA Toolkit 9.1 (Dec 2017)
    CUDA Toolkit 9.0 (Sept 2017)
    CUDA Toolkit 8.0 (Feb 2017)
    CUDA Toolkit 7.5 (Sept 2015)
    CUDA Toolkit 7.0 (March2015)
    CUDA Toolkit 6.5 (August 2014)
    CUDA Toolkit 6.0 (April 2014)
    CUDA Toolkit 5.5 (July 2013)
    CUDA Toolkit 5.0 (Oct 2012)
    CUDA Toolkit 4.2 (April 2012)
    CUDA Toolkit 4.1 (Jan 2012)
    CUDA Toolkit 4.0 (May 2011)
    CUDA Toolkit 3.2 (Nov 2010)
    CUDA Toolkit 3.1 (June 2010)
    CUDA Toolkit 3.0 (March 2010)

CUDA 5.5 corresponds to OptiX 3.5 setting these as Opticks minimum requirements. 
Should roughly correspond to era of compute capability 3.0 GPUs.


OSX Commandline User Switching and CUDA 
------------------------------------------

With "SystemPreferences>EnergySaver[Automatic graphics switching" enabled
(usual approach, as otherwise tend to get out of GPU memory often)
no CUDA devices are seen.::

    delta:build simon$ cudaGetDevicePropertiesTest 
    CUDA Device Query...target -1 
    There are 0 CUDA devices.
    0


With automatic graphics switching disabled (ie with the discrete GPU in use all the time)
CUDA sees the device::

    delta:build simon$ cudaGetDevicePropertiesTest 
    CUDA Device Query...target -1 
    There are 1 CUDA devices.

    CUDA Device #0
    Major revision number:         3
    Minor revision number:         0
    Name:                          GeForce GT 750M
    Total global memory:           2147024896
    Total shared memory per block: 49152
    Total registers per block:     65536
    Warp size:                     32
    Maximum memory pitch:          2147483647
    Maximum threads per block:     1024
    Maximum dimension 0 of block:  1024
    Maximum dimension 1 of block:  1024
    Maximum dimension 2 of block:  64
    Maximum dimension 0 of grid:   2147483647
    Maximum dimension 1 of grid:   65535
    Maximum dimension 2 of grid:   65535
    Clock rate:                    925500
    Total constant memory:         65536
    Texture alignment:             512
    Concurrent copy and execution: Yes
    Number of multiprocessors:     2
    Kernel execution timeout:      Yes
    30
    delta:build simon$ 
    delta:build simon$ vip


CUDA from console mode : is kinda similar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/index.html#axzz4EUM2eoYO

Note: To run CUDA applications in console mode on MacBook Pro with both an integrated GPU and a discrete GPU, 
use the following settings before dropping to console mode:

* Uncheck System Preferences > Energy Saver > Automatic Graphic Switch
* Drag the Computer sleep bar to Never in System Preferences > Energy Saver



CUDA 7.5 Release Notes
------------------------

Thrust
~~~~~~

* http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Toolkit_Release_Notes.pdf

The version of Thrust included with CUDA 7.5 has been upgraded to Thrust v1.8.2. 
Note that CUDA 7.0 shipped with Thrust v1.8.1. 
A changelog of the bugs fixed in v1.8.2 can be found at

* https://github.com/thrust/thrust/blob/1.8.2/CHANGELOG.

cuda-gdb 
~~~~~~~~~~

Debugging GPGPU code using cuda-gdb is no longer supported on the Mac platform.


FindCUDA.cmake
---------------

* http://www.cmake.org/cmake/help/v3.0/module/FindCUDA.html

::

    simon:cudarap blyth$ mdfind -name FindCUDA.cmake

    /Developer/OptiX_301/SDK/CMake/FindCUDA.cmake
    /usr/local/env/cuda/OptiX_370b2_sdk/CMake/FindCUDA.cmake
    /Developer/OptiX_370b2/SDK/CMake/FindCUDA.cmake
    /Developer/OptiX_380/SDK/CMake/FindCUDA.cmake
    /usr/local/env/cuda/OptiX_380_sdk/CMake/FindCUDA.cmake

    /opt/local/share/cmake-3.4/Modules/FindCUDA.cmake

    /usr/local/env/graphics/hrt/cmake/Modules/FindCUDA.cmake
    /usr/local/env/graphics/photonmap/CMake/FindCUDA.cmake
    /usr/local/env/optix/macrosim/macrosim_tracer/CMake/FindCUDA.cmake






Using CUDA GPU from Console Mode on Macbook Pro
-------------------------------------------------

* http://developer.download.nvidia.com/compute/cuda/6_0/rel/docs/CUDA_Getting_Started_Mac.pdf

To run CUDA applications in console mode on MacBook Pro with both an integrated GPU and a discrete GPU, 
use the following settings before dropping to console mode:

1. Uncheck System Preferences > Energy Saver > Automatic Graphic Switch
2. Drag the Computer sleep bar to Never in System Preferences > Energy Saver


To login with ">console" mode:

* logout 
* enter ">console" in the usename panel, press return : will drop to console login prompt 
* after running commands, ctrl-D to return to GUI login  


CUDA Debugging
----------------

* http://on-demand.gputechconf.com/gtc/2013/presentations/S3037-S3038-Debugging-CUDA-Apps-Linux-Mac.pdf


Installation : cuda-pkg-install  (June 29, 2015)
-------------------------------------------------

* See cudainstall- for version verifications

::

    cuda-
    cuda-vi            # set the version
    cuda-get           # download the pkg 
    cuda-pkg-install   # run GUI installer

PKG installer gives the below options, selected them all::

                  action       size
   CUDA Driver       Upgrade      83.5 MB
   CUDA Toolkit      Upgrade       1.3 GB
   CUDA Samples      Upgrade     199.3 MB

Installing them all took less than a minute.


Installing/building samples
----------------------------

::

    simon:cuda blyth$ cuda-
    simon:cuda blyth$ cuda-samples-install
    Copying samples to /usr/local/env/cuda/NVIDIA_CUDA-7.0_Samples now...
    Finished copying samples.

    simon:NVIDIA_CUDA-7.0_Samples blyth$ cuda-samples-make
    /Developer/NVIDIA/CUDA-7.0/bin/nvcc 
              -ccbin clang++ 
              -I../../common/inc  
              -m64 
              -Xcompiler -arch -Xcompiler x86_64  
              -gencode arch=compute_20,code=sm_20 
              -gencode arch=compute_30,code=sm_30  
              -gencode arch=compute_35,code=sm_35 
              -gencode arch=compute_37,code=sm_37 
              -gencode arch=compute_50,code=sm_50 
              -gencode arch=compute_52,code=sm_52 
              -gencode arch=compute_52,code=compute_52
              -o asyncAPI.o 
              -c asyncAPI.cu
     ...

     /Developer/NVIDIA/CUDA-7.0/bin/nvcc 
                -ccbin clang++   
                -m64  
                -Xcompiler -arch -Xcompiler x86_64  
                -Xlinker -rpath -Xlinker /Developer/NVIDIA/CUDA-7.0/lib 
                -gencode arch=compute_35,code=sm_35 
                -gencode arch=compute_37,code=sm_37 
                -gencode arch=compute_50,code=sm_50 
                -gencode arch=compute_52,code=sm_52 
                -gencode arch=compute_52,code=compute_52 
                -o cdpLUDecomposition 
                 cdp_lu.o cdp_lu_main.o dgetf2.o dgetrf.o dlaswp.o  
                -lcublas -lcublas_device -lcudadevrt


Hmm the build takes as long time, building for many arch/code 




nvcc help
------------

:: 

     nvcc -h

        --compile                                  (-c)                              
                Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file.

        --device-c                                 (-dc)                             
                Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
                relocatable device code.  It is equivalent to '--relocatable-device-code=true
                --compile'.

        --device-w                                 (-dw)                             
                Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
                executable device code.  It is equivalent to '--relocatable-device-code=false
                --compile'.

        --device-link                              (-dlink)                          
                Link object files with relocatable device code and .ptx/.cubin/.fatbin files
                into an object file with executable device code, which can be passed to the
                host linker.

        --link  (-link)                           
                This option specifies the default behavior: compile and link all inputs.

        --lib   (-lib)                            
                Compile all inputs into object files (if necessary) and add the results to
                the specified output library file.


       --compiler-bindir <path>                   (-ccbin)                          
            Specify the directory in which the host compiler executable resides.  The
            host compiler executable name can be also specified to ensure that the correct
            host compiler is selected.  In addition, driver prefix options ('--input-drive-prefix',
            '--dependency-drive-prefix', or '--drive-prefix') may need to be specified,
            if nvcc is executed in a Cygwin shell or a MinGW shell on Windows.


       --std {c++11}                              (-std)                            
            Select a particular C++ dialect.  The only value currently supported is "c++11".
            Enabling C++11 mode also turns on C++11 mode for the host compiler.
            Allowed values for this option:  'c++11'.

      --compiler-options <options>,...           (-Xcompiler)                      
            Specify options directly to the compiler/preprocessor.

      --linker-options <options>,...             (-Xlinker)                        
            Specify options directly to the host linker.

      --generate-code <specification>,...        (-gencode)                        
            This option provides a generalization of the '--gpu-architecture=<arch> --gpu-code=<code>,
            ...' option combination for specifying nvcc behavior with respect to code
            generation.  Where use of the previous options generates code for different
            'real' architectures with the PTX for the same 'virtual' architecture, option
            '--generate-code' allows multiple PTX generations for different 'virtual'
            architectures.  In fact, '--gpu-architecture=<arch> --gpu-code=<code>,
            ...' is equivalent to '--generate-code arch=<arch>,code=<code>,...'.
            '--generate-code' options may be repeated for different virtual architectures.
            Allowed keywords for this option:  'arch','code'.





tips to free up GPU memory
---------------------------

#. minimise the number of apps and windows running
#. put the machine to sleep and go and have a coffee

#. dont use dedicated GPU mode, this avoids most GPU memory concerns
   presumably as it reduces prsssure on the GPU



CUDA Constant Memory : small size 64K
----------------------------------------

::

    simon:~ blyth$ cuda-;cuda-deviceQuery | grep constant
      Total amount of constant memory:               65536 bytes


* http://cuda-programming.blogspot.tw/2013/01/what-is-constant-memory-in-cuda.html?m=1

Whether it is advantageous depends on access pattern, if every thread in warp (half warp?)
is accessing the same address of constant memory simulataneously it is very performant 
if not the accesses are serialized. 

* Think OpenGL uniforms

* So: do not early exit from loop over contant memory array when doing a lookup check ?   



CUDA Driver and Runtime API interop
------------------------------------

* http://stackoverflow.com/questions/20539349/cuda-runtime-api-and-dynamic-kernel-definition

  * http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER


Bit packing with CUDA vector types
-------------------------------------

* http://nvlabs.github.io/cub/index.html


nvcc compilation for newer architectures
------------------------------------------

Some PTX instructions are only supported on devices of higher compute
capabilities. For example, warp shuffle instructions are only supported on
devices of compute capability 3.0 and above. The -arch compiler option
specifies the compute capability that is assumed when compiling C to PTX code.
So, code that contains warp shuffle, for example, must be compiled with
-arch=sm_30 (or higher).

PTX code produced for some specific compute capability can always be compiled
to binary code of greater or equal compute capability.


::

    delta:~ blyth$ nvcc -o /usr/local/env/cuda/texture/cuda_texture_object -arch=sm_30 /Users/blyth/env/cuda/texture/cuda_texture_object.cu
    delta:~ blyth$ nvcc -o /usr/local/env/cuda/texture/cuda_texture_object  /Users/blyth/env/cuda/texture/cuda_texture_object.cu
    /Users/blyth/env/cuda/texture/cuda_texture_object.cu(10): error: type name is not allowed

    /Users/blyth/env/cuda/texture/cuda_texture_object.cu(10): warning: expression has no effect

    1 error detected in the compilation of "/tmp/tmpxft_00017bac_00000000-6_cuda_texture_object.cpp1.ii".
    delta:~ blyth$ 


syslog : Understanding XID Errors
------------------------------------

* http://docs.nvidia.com/deploy/xid-errors/index.html



deviceQuery
-------------

::

    delta:~ blyth$ cuda-samples-bin-deviceQuery
    running /usr/local/env/cuda/NVIDIA_CUDA-5.5_Samples/bin/x86_64/darwin/release/deviceQuery
    /usr/local/env/cuda/NVIDIA_CUDA-5.5_Samples/bin/x86_64/darwin/release/deviceQuery Starting...

     CUDA Device Query (Runtime API) version (CUDART static linking)

    Detected 1 CUDA Capable device(s)

    Device 0: "GeForce GT 750M"
      CUDA Driver Version / Runtime Version          5.5 / 5.5
      CUDA Capability Major/Minor version number:    3.0
      Total amount of global memory:                 2048 MBytes (2147024896 bytes)
      ( 2) Multiprocessors, (192) CUDA Cores/MP:     384 CUDA Cores
      GPU Clock rate:                                926 MHz (0.93 GHz)
      Memory Clock rate:                             2508 Mhz
      Memory Bus Width:                              128-bit
      L2 Cache Size:                                 262144 bytes
      Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
      Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
      Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
      Total amount of constant memory:               65536 bytes
      Total amount of shared memory per block:       49152 bytes
      Total number of registers available per block: 65536
      Warp size:                                     32
      Maximum number of threads per multiprocessor:  2048
      Maximum number of threads per block:           1024
      Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
      Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
      Maximum memory pitch:                          2147483647 bytes
      Texture alignment:                             512 bytes
      Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
      Run time limit on kernels:                     Yes
      Integrated GPU sharing Host Memory:            No
      Support host page-locked memory mapping:       Yes
      Alignment requirement for Surfaces:            Yes
      Device has ECC support:                        Disabled
      Device supports Unified Addressing (UVA):      Yes
      Device PCI Bus ID / PCI location ID:           1 / 0
      Compute Mode:
         < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

    deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 5.5, CUDA Runtime Version = 5.5, NumDevs = 1, Device0 = GeForce GT 750M
    Result = PASS



bandwidthTest
--------------

::

    delta:~ blyth$ cuda-samples-bin-bandwidthTest
    running /usr/local/env/cuda/NVIDIA_CUDA-5.5_Samples/bin/x86_64/darwin/release/bandwidthTest
    [CUDA Bandwidth Test] - Starting...
    Running on...

     Device 0: GeForce GT 750M
     Quick Mode

     Host to Device Bandwidth, 1 Device(s)
     PINNED Memory Transfers
       Transfer Size (Bytes)    Bandwidth(MB/s)
       33554432         1566.0

     Device to Host Bandwidth, 1 Device(s)
     PINNED Memory Transfers
       Transfer Size (Bytes)    Bandwidth(MB/s)
       33554432         3182.6

     Device to Device Bandwidth, 1 Device(s)
     PINNED Memory Transfers
       Transfer Size (Bytes)    Bandwidth(MB/s)
       33554432         17074.5

    Result = PASS
    delta:~ blyth$



Profiling
-----------

nvvp and nsight
~~~~~~~~~~~~~~~~~~

::

    delta:doc blyth$ nvvp
    Unable to find any JVMs matching architecture "i386".
    No Java runtime present, try --request to install.
    No Java runtime present, requesting install.

    delta:doc blyth$ which nsight
    /Developer/NVIDIA/CUDA-5.5/bin/nsight
    delta:doc blyth$ nsight
    Unable to find any JVMs matching architecture "i386".
    No Java runtime present, try --request to install.
    No Java runtime present, requesting install.
    delta:doc blyth$ 




FindCUDA.cmake
---------------

* http://www.cmake.org/cmake/help/v3.0/module/FindCUDA.html

* http://stackoverflow.com/questions/13683575/cuda-5-0-separate-compilation-of-library-with-cmake

::

    simon:ggeoview blyth$ ggeoview-cmake
    Requested CUDA version 7.0, but found unacceptable version 5.5
    CMake Error at /opt/local/share/cmake-3.2/Modules/FindPackageHandleStandardArgs.cmake:138 (message):
      Could NOT find CUDA (missing: _cuda_version_acceptable) (Required is at
      least version "7.0")
    Call Stack (most recent call first):
      /opt/local/share/cmake-3.2/Modules/FindPackageHandleStandardArgs.cmake:374 (_FPHSA_FAILURE_MESSAGE)
      /Developer/OptiX/SDK/CMake/FindCUDA.cmake:889 (find_package_handle_standard_args)
      CMakeLists.txt:28 (find_package)


    -- Configuring incomplete, errors occurred!
    See also "/usr/local/env/graphics/ggeoview.build/CMakeFiles/CMakeOutput.log".
    simon:ggeoview blyth$ 






CUDA CMAKE Mavericks
-----------------------

Informative explanation of Mavericks CUDA challenges wrt compiler flags...
 
https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks

See::

   cudawrap-
   thrustrap-
   thrusthello-
   thrust-



CUDA OSX libc++
-----------------



CUDA Release Notes
-------------------

* http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

On Mac OS X, libc++ is supported with XCode 5.x. Command-line option -Xcompiler
-stdlib=libstdc++ is no longer needed when invoking NVCC. Instead, NVCC uses
the default library that Clang chooses on Mac OS X. Users are still able to
choose between libc++ and libstdc++ by passing -Xcompiler -stdlib=libc++ or
-Xcompiler -stdlib=libstdc++ to NVCC.

The Runtime Compilation library (nvrtc) provides an API to compile CUDA-C++
device source code at runtime. The resulting compiled PTX can be launched on a
GPU using the CUDA Driver API. More details can be found in the libNVRTC User
Guide.

Added C++11 support. The new nvcc flag -std=c++11 turns on C++11 features in
the CUDA compiler as well as the host compiler and linker. The flag is
supported by host compilers Clang and GCC versions 4.7 and newer. In addition,
any C++11 features that are enabled by default by a supported host compiler are
also allowed to be used in device code. Please see the CUDA Programming Guide
for further details.



FUNCTIONS
---------


cuda-get

EOU
}

cuda-export(){
   echo -n
}

cuda-nvcc-flags(){
    case $NODE_TAG in 
       D) echo -ccbin /usr/bin/clang --use_fast_math ;;
       *) echo --use_fast_math ;; 
    esac 
}


#cuda-version(){      echo ${CUDA_VERSION:-5.5} ; }
#cuda-version(){      echo ${CUDA_VERSION:-7.0} ; }
cuda-version(){      echo ${CUDA_VERSION:-9.1} ; }
cuda-download-dir(){ echo $(local-base)/env/cuda ; }


cuda-dir()
{       
   case $(uname) in 
       Linux) echo /usr/local/cuda-$(cuda-version) ;;
      Darwin) echo /Developer/NVIDIA/CUDA-$(cuda-version) ;; 
    MINGW64*) echo /tmp ;;
   esac
}

cuda-prefix(){       echo $(cuda-dir) ; }
cuda-edir(){         echo $(opticks-home)/cuda ; }
cuda-idir(){         echo $(cuda-dir)/include ; }

cuda-writable-dir(){ 
  case $(uname) in
    Linux) echo $(local-base)/env/cuda ;;
    Darwin) echo /usr/local/epsilon/cuda ;;
  esac
} 
cuda-samples-dir(){  echo $(cuda-writable-dir)/NVIDIA_CUDA-$(cuda-version)_Samples ; }
cuda-samples-find(){ 
   find $(cuda-samples-dir) -type f -exec grep -${2:-l} ${1:-cuda_gl_interop.h} {} \;  
}


cuda-cd(){           cd $(cuda-dir); }
cuda-ecd(){          cd $(cuda-edir); }
cuda-icd(){          cd $(cuda-idir); }
cuda-dcd(){          cd $(cuda-download-dir); }
cuda-wcd(){          cd $(cuda-writable-dir); }
cuda-samples-cd(){   cd $(cuda-samples-dir) ; }

cuda-find(){ find $(cuda-idir) -name '*.h' -exec grep -H ${1:-cudaGraphics} {} \; ; }

cuda-libdir(){
   case $NODE_TAG in 
      D) echo /usr/local/cuda/lib ;;
      G1) echo /usr/local/cuda/lib64 ;;
   esac
}

cuda-path(){
    local dir=$(cuda-dir)
    [ ! -d $dir ] && return 1
    export PATH=$dir/bin:$PATH
    local libdir=$(cuda-libdir)

    if [ "$(uname)" == "Darwin" ]; then 
       export DYLD_LIBRARY_PATH=$libdir:$DYLD_LIBRARY_PATH      # not documented ??? links from this to $dir/lib
    else
       export LD_LIBRARY_PATH=$libdir:$LD_LIBRARY_PATH
    fi
    # these are not seen by the pycuda build
}


cuda-url(){ echo $(cuda-url-$(uname)) ; }
cuda-url-Darwin(){
   case $(cuda-version) in 
       5.5) echo http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/cuda-mac-5.5.28_10.9_64.pkg ;;
       7.0) echo http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.29_mac.pkg ;;
   esac
}
cuda-pkg(){  echo $(basename $(cuda-url)) ; }
cuda-pkgpath(){ echo $(cuda-download-dir)/$(cuda-pkg) ; }
cuda-get(){
   local dir=$(cuda-download-dir) &&  mkdir -p $dir && cd $dir
   local url=$(cuda-url-$(uname))
   local pkg=$(cuda-pkg)
   [ -d $(cuda-dir) ] && echo $msg CUDA is already installed at $(cuda-dir) && return 0 
   [ ! -f "$pkg" ] && curl -L -O $url 
}

cuda-pkg-install(){
   local pkg=$(cuda-pkgpath)
   open $pkg   
}

cuda-pkgpath-lsbom()
{
   lsbom $(pkgutil --bom $(cuda-pkgpath))
}

cuda-osx-pkginfo(){         installer -pkginfo -pkg $(dirname $(cuda-download-dir))/$(cuda-pkg) ; }
cuda-osx-getting-started(){ open $(cuda-dir)/doc/html/cuda-getting-started-guide-for-mac-os-x/index.html ; }
cuda-guide(){               open $(cuda-dir)/doc/html/cuda-c-programming-guide/index.html ; }
cuda-doc(){                 open $(cuda-dir)/doc/html/index.html ; }
cuda-osx-kextstat(){        kextstat | grep -i cuda ; }

cuda-pdf-(){ echo $(cuda-dir)/doc/pdf/CUDA_C_Programming_Guide.pdf ; }
cuda-pdf(){ open $(cuda-pdf-) ; }




cuda-samples-install(){
   local iwd=$PWD
   local dir=$(cuda-writable-dir)
   [ ! -d "$dir" ] && mkdir -p $dir

   cuda-install-samples-$(cuda-version).sh $dir
}

cuda-samples-install-skipping-doc(){
   local iwd=$PWD
   local dir=$(cuda-writable-dir)
   [ ! -d "$dir" ] && mkdir -p $dir
  
   # cuda-install-samples-5.5.sh $dir
   # have to do it manually to skip the doc folder which goes over afs quota

   local src=$(dirname $(dirname $(which cuda-install-samples-$(cuda-version).sh)))/samples
   local dst=$(cuda-samples-dir)
   local path

   cd $src
   local name
   ls -1 $src | while read path ; do
      name=$(basename $path)
      if [ "$name" == "doc" ]; then
         echo skip $path
      else
         cp -R $path $dst/
      fi
   done
   cd $iwd
}

cuda-samples-make(){
   cuda-samples-cd
   make $*
}


cuda-samples-bin-dir(){ echo $(cuda-samples-dir)/bin/$(uname -m)/$(uname | tr '[:upper:]' '[:lower:]')/release ; }
cuda-samples-bin-cd(){  cd $(cuda-samples-bin-dir) ; }

cuda-samples-bin-run(){  
    local bin=$(cuda-samples-bin-dir)/$1 
    shift   
    local cmd="$bin $*"
    [ ! -x $bin ] && echo $msg bin $bin not found OR not executable && return 1 
    [ -x $bin ] && echo $msg running $cmd
    eval $cmd
}

cuda-samples-bin-deviceQuery(){    cuda-samples-bin-run deviceQuery $* ; }
cuda-samples-bin-bandwidthTest(){  cuda-samples-bin-run bandwidthTest $* ; }
cuda-samples-bin-smokeParticles(){ cuda-samples-bin-run smokeParticles $* ; }
cuda-samples-bin-fluidsGL(){       cuda-samples-bin-run fluidsGL $* ; }

cuda-deviceQuery(){ cuda-samples-bin-run deviceQuery $* ; } 
