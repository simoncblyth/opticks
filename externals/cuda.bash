##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

cuda-source(){   echo $BASH_SOURCE ; }
cuda-vi(){       vi $(cuda-source) ; }
cuda-env(){      olocal- ; cuda-path ; }
cuda-usage(){ cat << EOU

CUDA
======

See Also
---------


* FOR MORE RECENT INSTALL SEE cudalin-

* cudatoolkit-
* cudamac-
* env-;cudatex-



cudaMemcpyToArray deprecated
------------------------------

* https://forums.developer.nvidia.com/t/cudamemcpytoarray-is-deprecated/71385/9

::

    data/blyth/junotop/opticks/qudarap/QTex.cc: In member function ‘void QTex<T>::uploadToArray()’:
    /data/blyth/junotop/opticks/qudarap/QTex.cc:121:5: warning: ‘cudaError_t cudaMemcpyToArray(cudaArray_t, size_t, size_t, const void*, size_t, cudaMemcpyKind)’ is deprecated (declared at /usr/local/cuda/include/cuda_runtime_api.h:6595) [-Wdeprecated-declarations]
         cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind );
         ^



128bit uint ?
----------------

* https://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda
* https://github.com/curtisseizert/CUDA-uint128


CUdeviceptr
------------

* http://www.cudahandbook.com/2013/08/why-does-cuda-cudeviceptr-use-unsigned-int-instead-of-void/

* 
* uintptr_t is an unsigned integer type that is capable of storing a data pointer.

::

    void *p;
    CUdeviceptr dptr;
    p = (void *) (uintptr_t) dptr;
    dptr = (CUdeviceptr) (uintptr_t) p;

::

     // uintptr_t is an unsigned integer type that is capable of storing a data pointer.
     // CUdeviceptr is typedef to unsigned long lonh 
     std::cout << "                          d_psd.aabb " <<                         d_psd.aabb << std::endl ;   
     std::cout << "               (uintptr_t)d_psd.aabb " <<              (uintptr_t)d_psd.aabb << std::endl ;   
     std::cout << "  (CUdeviceptr)(uintptr_t)d_psd.aabb " << (CUdeviceptr)(uintptr_t)d_psd.aabb << std::endl ; 



Alignment
-----------

* https://stackoverflow.com/questions/12778949/cuda-memory-alignment

::

    #if defined(__CUDACC__) // NVCC
       #define MY_ALIGN(n) __align__(n)
    #elif defined(__GNUC__) // GCC
       #define MY_ALIGN(n) __attribute__((aligned(n)))
    #elif defined(_MSC_VER) // MSVC
       #define MY_ALIGN(n) __declspec(align(n))
    #else
       #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
    #endif

    struct MY_ALIGN(16) pt { int i, j, k; }


In this case let's say we choose 16-byte alignment. On a 32-bit machine, the
pointer takes 4 bytes, so the struct takes 20 bytes. 16-byte alignment will
waste 16 * (ceil(20/16) - 1) = 12 bytes per struct. On a 64-bit machine, it
will waste only 8 bytes per struct, due to the 8-byte pointer. We can reduce
the waste by using MY_ALIGN(8) instead. The tradeoff will be that the hardware
will have to use 3 8-byte loads instead of 2 16-byte loads to load the struct
from memory. If you are not bottlenecked by the loads, this is probably a
worthwhile tradeoff. Note that you don't want to align smaller than 4 bytes for
this struct.



Samples
---------

* https://github.com/NVIDIA/cuda-samples


CUDA VM
----------

* https://stackoverflow.com/questions/11631191/why-does-the-cuda-runtime-reserve-80-gib-virtual-memory-upon-initialization

talonmies
    Nothing to do with scratch space, it is the result of the addressing system
    that allows unified andressing and peer to peer access between host and
    multiple GPUs. The CUDA driver registers all the GPU(s) memory + host memory in
    a single virtual address space using the kernel's virtual memory system. It
    isn't actually memory consumption, per se, it is just a "trick" to map all the
    available address spaces into a linear virtual space for unified addressing.



CUDA gcc 9 : CUDA_HOST_COMPILER=/usr/bin/gcc-8 
------------------------------------------------

* https://ingowald.blog/2020/05/14/quick-note-on-cuda-owl-on-ubuntu-20/
* https://forums.developer.nvidia.com/t/cmake-file-bug-in-optix-5-x-can-cause-incorrectly-formatted-nvcc-command/63076


CUDA Multithreading
---------------------

* https://docs.nvidia.com/cuda/cuda-samples/index.html#cudaopenmp

::

    cuda-samples-cd 0_Simple/simpleMPI
    cuda-samples-cd 0_Simple/simpleCallback


    epsilon:simpleMPI blyth$ cuda-samples-find pthread.h
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/4_Finance/MonteCarloMultiGPU/multithreading.h
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/6_Advanced/interval/boost/config/platform/bsd.hpp
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/6_Advanced/threadMigration/threadMigration.cpp
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/common/inc/multithreading.h
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/0_Simple/UnifiedMemoryStreams/UnifiedMemoryStreams.cu
    /usr/local/epsilon/cuda/NVIDIA_CUDA-9.1_Samples/0_Simple/simpleCallback/multithreading.h
    epsilon:simpleMPI blyth$ 



CUDA learning refs
---------------------

* https://streamhpc.com/blog/2017-01-24/many-threads-can-run-gpu/

CUDA Application Design and Development, Chapter 1 - First Programs and How to Think in CUDA

* https://www.sciencedirect.com/science/article/pii/B978012388426800001X

* https://www.sciencedirect.com/topics/computer-science/amdahls-law



CUDA envvars
-------------

* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars

CUDA_VISIBLE_DEVICES : A comma-separated sequence of GPU identifiers 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU identifiers are given as integer indices or as UUID strings. GPU UUID
strings should follow the same format as given by nvidia-smi, such as
GPU-8932f937-d72c-4106-c12f-20bd9faed9f6.

Only the devices whose index is present in the sequence are visible to CUDA
applications and they are enumerated in the order of the sequence. If one of
the indices is invalid, only the devices whose index precedes the invalid index
are visible to CUDA applications.

CUDA_DEVICE_ORDER : FASTEST_FIRST, PCI_BUS_ID, (default is FASTEST_FIRST) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FASTEST_FIRST causes CUDA to guess which device is fastest using a simple
heuristic, and make that device 0, leaving the order of the rest of the devices
unspecified. PCI_BUS_ID orders devices by PCI bus ID in ascending order. 


CURAND
--------

* http://richiesams.blogspot.com/2015/03/creating-randomness-and-acummulating.html



Measuring Bandwidth
-----------------------

* https://devtalk.nvidia.com/default/topic/382019/cuda-programming-and-performance/how-to-get-peak-rate-with-simple-opeartion-question-about-performance-optimization/#entry290441


CUDA CMake : Building using Modern CMake Native CUDA Support 
-------------------------------------------------------------

::

    project(${name} VERSION 0.1.0 LANGUAGES CXX CUDA )  


* https://stackoverflow.com/questions/50583886/rosetta-for-switching-to-native-cmake-cuda-support

* https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake/

  * CMake 3.8 makes CUDA C++ an intrinsically supported language 
  * with example 

  * BUT its doesnt work for me 

::

    epsilon:recon blyth$ DYLD_LIBRARY_PATH=/tmp/blyth/intro_to_cuda/recon/install/lib /tmp/blyth/intro_to_cuda/recon/install/lib/ReconTest
    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: device free failed: CUDA driver version is insufficient for CUDA runtime version
    Abort trap: 6
       

Note no CUDA in above languages list as the very new native CUDA support 
that would switch on does not work for me, at runtime it gives::

    CUDA driver version is insufficient for CUDA runtime version

However the old FindCUDA.cmake approach works







CuPP
-----

* https://www.jensbreitbart.de/pdf/frameworkeasyCUDAintegration.pdf
* ~/opticks_refs/CuPP_frameworkeasyCUDAintegration.pdf


Intros
----------

* https://devblogs.nvidia.com/even-easier-introduction-cuda/
* https://devblogs.nvidia.com/unified-memory-cuda-beginners/


GPU Compute Capability
------------------------

* https://developer.nvidia.com/cuda-gpus#collapse5

* OptiX 6 (Feb 2019, with RTX support) is raising the bar 
  from Compute Capability 3.0(Kepler) to 5.0(Maxwell)


Tesla : Data Center
~~~~~~~~~~~~~~~~~~~~

=================  ======
Data Center GPU     CC
=================  ======
Tesla T4            7.5
Tesla V100          7.0
Tesla P100          6.0
Tesla P40           6.1
Tesla P4            6.1
Tesla M60           5.2
Tesla M40           5.2
-----------------  ------
Tesla K80           3.7
Tesla K40           3.5
Tesla K20           3.5
Tesla K10           3.0
=================  ======


GeForce : Desktop 
~~~~~~~~~~~~~~~~~~~~~~

=======================  ====================
GPU                       Compute Capability
=======================  ====================
NVIDIA TITAN RTX         7.5
Geforce RTX 2080 Ti      7.5
Geforce RTX 2080         7.5
Geforce RTX 2070         7.5
Geforce RTX 2060         7.5
NVIDIA TITAN V           7.0
NVIDIA TITAN Xp          6.1
NVIDIA TITAN X           6.1
GeForce GTX 1080 Ti      6.1
GeForce GTX 1080         6.1
GeForce GTX 1070         6.1
GeForce GTX 1060         6.1
GeForce GTX 1050         6.1
GeForce GTX TITAN X      5.2
GeForce GTX 980 Ti       5.2
GeForce GTX 980          5.2
GeForce GTX 970          5.2
GeForce GTX 960          5.2
GeForce GTX 950          5.2
GeForce GTX 750 Ti       5.0
GeForce GTX 750          5.0
-----------------------  --------------------
GeForce GTX 780 Ti       3.5
GeForce GTX 780          3.5
GeForce GTX 770          3.0
GeForce GTX 760          3.0
GeForce GTX TITAN Z      3.5
GeForce GTX TITAN Black  3.5
GeForce GTX TITAN        3.5
GeForce GTX 690          3.0
GeForce GTX 680          3.0
GeForce GTX 670          3.0
GeForce GTX 660 Ti       3.0
GeForce GTX 660          3.0
GeForce GTX 650 Ti BOOST 3.0
GeForce GTX 650 Ti       3.0
GeForce GTX 650          3.0
=======================  ====================







dynamic parallelism : series by  Andy Adinets
-------------------------------------------------

CUDA 5.0 introduced Dynamic Parallelism (Compute Capability 3.5 or higher), 
which makes it possible to launch kernels from threads running on the device; 
threads can launch more threads. An application can launch a coarse-grained 
kernel which in turn launches finer-grained kernels to do work where needed.



* https://devblogs.nvidia.com/introduction-cuda-dynamic-parallelism/

  Multi-level parallelism example that avoids waisting compute in the black zone
  inside the Mandelbrot set (Mariani-Silver Algorithm) 

  Dynamic Parallelism uses the CUDA Device Runtime library (cudadevrt), 
  a subset of CUDA Runtime API callable from device code.


* https://devblogs.nvidia.com/cuda-dynamic-parallelism-api-principles/

  practicalities of nested launches

* https://devblogs.nvidia.com/a-cuda-dynamic-parallelism-case-study-panda/
* http://on-demand.gputechconf.com/gtc/2014/presentations/S4499-gpus-for-online-track-reconstruction.pdf 


how is CUDA memory managed ?
-----------------------------

* https://stackoverflow.com/questions/8684770/how-is-cuda-memory-managed


cuda uninstallers
-------------------

* http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#uninstall

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


* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

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


CUDA 9.1 SAMPLES
------------------

::

    epsilon:local blyth$ sudo cp -r /Developer/NVIDIA/CUDA-9.1/samples cuda_9_1_samples
    epsilon:local blyth$ sudo chown -R blyth:staff cuda_9_1_samples




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






cuda-edit(){ vi /opt/local/share/cmake-3.11/Modules/FindCUDA.cmake ; }

cuda-uninstall-notes(){ cat << EON

Uninstall CUDA
================

* http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#uninstall

Below lists summarize the manifests.

uninstall driver
------------------

/usr/local/bin/.cuda_driver_uninstall_manifest_do_not_delete.txt

    /Library/Extensions/CUDA.kext
    /Library/Frameworks/CUDA.framework
    /Library/LaunchAgents/com.nvidia.CUDASoftwareUpdate.plist
    /Library/LaunchDaemons/com.nvidia.cuda.launcher.plist
    /Library/LaunchDaemons/com.nvidia.cudad.plist
    /Library/PreferencePanes/CUDA Preferences.prefPane 
    /usr/local/cuda/lib/libcuda.dylib 

Before uninstall::

    epsilon:~ blyth$ ll /Library/Extensions/
    total 0
    drwxr-xr-x   3 root  wheel    96 Aug 21  2013 ArcMSR.kext
    drwxr-xr-x   3 root  wheel    96 Sep  1  2013 CalDigitHDProDrv.kext
    drwxr-xr-x   3 root  wheel    96 Jun 13  2014 ACS6x.kext
    drwxr-xr-x   3 root  wheel    96 Aug 15  2014 HighPointIOP.kext
    drwxr-xr-x   3 root  wheel    96 Aug 15  2014 HighPointRR.kext
    drwxr-xr-x   3 root  wheel    96 Jun 28  2016 ATTOCelerityFC8.kext
    drwxr-xr-x   3 root  wheel    96 Jun 28  2016 ATTOExpressSASHBA2.kext
    drwxr-xr-x   3 root  wheel    96 Jun 28  2016 ATTOExpressSASRAID2.kext
    drwxr-xr-x   3 root  wheel    96 Mar 31  2017 PromiseSTEX.kext
    drwxr-xr-x   3 root  wheel    96 Aug 22  2017 SoftRAID.kext
    drwxr-xr-x   3 root  wheel    96 Dec 20 13:05 CUDA.kext
    drwxr-xr-x+ 66 root  wheel  2112 Mar 30 22:56 ..
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:40 NVDAResmanTeslaWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:40 NVDANV50HalTeslaWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:40 GeForceTeslaWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:57 NVDAStartupWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:57 NVDAResmanWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:57 NVDAGP100HalWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:57 NVDAGM100HalWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:57 NVDAGK100HalWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:57 NVDAGF100HalWeb.kext
    drwxr-xr-x   3 root  wheel    96 Mar 30 23:57 GeForceWeb.kext
    drwxr-xr-x  23 root  wheel   736 Mar 31 19:19 .
    epsilon:~ blyth$ 

    epsilon:~ blyth$ ll /Library/Frameworks/
    total 0
    drwxr-xr-x   7 blyth  wheel   224 Mar 31  2015 SDL2.framework
    drwxr-xr-x   8 root   wheel   256 Apr 24  2015 Mono.framework
    lrwxr-xr-x   1 root   wheel    71 Mar 28 17:37 AEProfiling.framework -> ../../Applications/Motion.app/Contents/Frameworks/AEProfiling.framework
    lrwxr-xr-x   1 root   wheel    74 Mar 28 17:37 AERegistration.framework -> ../../Applications/Motion.app/Contents/Frameworks/AERegistration.framework
    lrwxr-xr-x   1 root   wheel    74 Mar 28 17:37 AudioMixEngine.framework -> ../../Applications/Motion.app/Contents/Frameworks/AudioMixEngine.framework
    lrwxr-xr-x   1 root   wheel    60 Mar 28 17:37 NyxAudioAnalysis.framework -> /System/Library/PrivateFrameworks/NyxAudioAnalysis.framework
    drwxr-xr-x   5 root   wheel   160 Mar 28 17:42 PluginManager.framework
    drwxr-xr-x@  8 root   wheel   256 Mar 28 20:37 iTunesLibrary.framework
    drwxr-xr-x+ 66 root   wheel  2112 Mar 30 22:56 ..
    drwxr-xr-x   7 root   wheel   224 Mar 31 19:19 CUDA.framework
    drwxr-xr-x  11 root   wheel   352 Mar 31 19:19 .
    epsilon:~ blyth$ 

    epsilon:~ blyth$ ll /Library/LaunchAgents/
    total 24
    lrwxr-xr-x   1 root  wheel   104 Jul 23  2015 com.oracle.java.Java-Updater.plist -> /Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Resources/com.oracle.java.Java-Updater.plist
    lrwxr-xr-x   1 root  wheel    66 Oct 17  2016 org.freedesktop.dbus-session.plist -> /opt/local/Library/LaunchAgents/org.freedesktop.dbus-session.plist
    -rw-r--r--   1 root  wheel   715 Oct 26  2016 org.macosforge.xquartz.startx.plist
    -rw-r--r--   1 root  wheel   734 Aug 11  2017 com.nvidia.CUDASoftwareUpdate.plist
    drwxr-xr-x+ 66 root  wheel  2112 Mar 30 22:56 ..
    -rw-r--r--   1 root  wheel   665 Mar 31 19:04 com.nvidia.nvagent.plist
    drwxr-xr-x   7 root  wheel   224 Mar 31 19:19 .
    epsilon:~ blyth$ 

    epsilon:~ blyth$ ll /Library/PreferencePanes/
    total 0
    lrwxr-xr-x   1 root  wheel   101 Jul 23  2015 JavaControlPanel.prefPane -> /Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/lib/deploy/JavaControlPanel.prefPane
    drwxr-xr-x   3 root  wheel    96 Dec 20 13:05 CUDA Preferences.prefPane
    drwxr-xr-x   3 root  wheel    96 Mar 14 03:24 NVIDIA Driver Manager.prefPane
    drwxr-xr-x+ 66 root  wheel  2112 Mar 30 22:56 ..
    drwxr-xr-x   5 root  wheel   160 Mar 31 19:19 .
    epsilon:~ blyth$ 

    epsilon:~ blyth$ ll /usr/local/cuda/lib/
    total 32
    -rwxr-xr-x   1 root  wheel  13568 Dec 20 13:05 libcuda.dylib
    lrwxr-xr-x   1 root  wheel     36 Dec 20 19:54 stubs -> /Developer/NVIDIA/CUDA-9.1/lib/stubs
    lrwxr-xr-x   1 root  wheel     45 Dec 20 19:54 libnvrtc.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libnvrtc.dylib
    lrwxr-xr-x   1 root  wheel     49 Dec 20 19:54 libnvrtc.9.1.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libnvrtc.9.1.dylib
    lrwxr-xr-x   1 root  wheel     54 Dec 20 19:54 libnvrtc-builtins.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libnvrtc-builtins.dylib
    lrwxr-xr-x   1 root  wheel     58 Dec 20 19:54 libnvrtc-builtins.9.1.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libnvrtc-builtins.9.1.dylib
    lrwxr-xr-x   1 root  wheel     50 Dec 20 19:54 libnvgraph_static.a -> /Developer/NVIDIA/CUDA-9.1/lib/libnvgraph_static.a
    lrwxr-xr-x   1 root  wheel     47 Dec 20 19:54 libnvgraph.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libnvgraph.dylib
    lrwxr-xr-x   1 root  wheel     51 Dec 20 19:54 libnvgraph.9.1.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libnvgraph.9.1.dylib
    lrwxr-xr-x   1 root  wheel     46 Dec 20 19:54 libnvblas.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libnvblas.dylib
    ...

uninstall toolkit
-------------------

/Developer/NVIDIA/CUDA-9.1/bin/.cuda_toolkit_uninstall_manifest_do_not_delete.txt
    /Developer/NVIDIA/CUDA-9.1 
    /usr/local/cuda
    /usr/local/cuda/lib/

uninstall samples
-------------------

/Developer/NVIDIA/CUDA-9.1/bin/.cuda_samples_uninstall_manifest_do_not_delete.txt
    /Developer/NVIDIA/CUDA-9.1/samples








EON
}

cuda-uninstall()
{
   cuda-uninstall-driver
   cuda-uninstall- cuda_toolkit
   cuda-uninstall- cuda_samples
}

cuda-uninstall-driver()
{
    cd /usr/local/bin
    $SUDO perl uninstall_cuda_drv.pl 
}


cuda-uninstall-()
{
    local comp=${1:-cuda_toolkit}

    [ $(uname) != "Darwin" ] && echo $msg not tested/checked on Linux yet && return

    local dir=$(cuda-dir)/bin
    [ ! -d "$dir" ] && echo $msg no dir $dir && return 

    cd $dir 
    $SUDO perl uninstall_cuda_$(cuda-version).pl  --manifest=.${comp}_uninstall_manifest_do_not_delete.txt
}


#cuda-prefix(){       echo $(cuda-dir) ; }
cuda-edir(){         echo $(opticks-home)/cuda ; }
cuda-idir(){         echo $(cuda-dir)/include ; }

cuda-writable-dir(){ 
  case $(uname) in
    Linux) echo $(local-base) ;;
    Darwin) echo /usr/local/epsilon/cuda ;;
  esac
} 

cuda-samples-name(){  echo NVIDIA_CUDA-$(cuda-version)_Samples ; }
cuda-samples-place(){
   local msg="=== $FUNCNAME :"
   local name=$(cuda-samples-name)
   local dir=$HOME/$name
   [ ! -d "$dir" ] && echo $msg no dir $dir && return

   local wrt=$(cuda-writable-dir)
   mkdir -p $wrt 

   local cmd="mv $dir $wrt/"
   echo $msg cmd \"$cmd\"
   local ans
   read -p "$msg enter Y to proceed : "

   [ "$ans" != "Y" ] && echo $msg skip && return
   eval $cmd
   
}
cuda-samples-dir(){  echo $(cuda-writable-dir)/$(cuda-samples-name) ; }
cuda-samples-find(){ 
   find $(cuda-samples-dir) -type f -exec grep -${2:-l} ${1:-cuda_gl_interop.h} {} \;  
}

cuda-samples-tex2D(){  cuda-samples-find tex2D ; }
cuda-samples-tex3D(){  cuda-samples-find tex3D ; }



cuda-cd(){           cd $(cuda-dir); }
cuda-ecd(){          cd $(cuda-edir); }
cuda-icd(){          cd $(cuda-idir); }
cuda-dcd(){          cd $(cuda-download-dir); }
cuda-wcd(){          cd $(cuda-writable-dir); }
cuda-samples-cd(){   cd $(cuda-samples-dir)/$1 ; }

cuda-find(){ find $(cuda-idir) -name '*.h' -exec grep -H ${1:-cudaGraphics} {} \; ; }

cuda-libdir(){
   case $NODE_TAG in 
      D) echo /usr/local/cuda/lib ;;
      G1) echo /usr/local/cuda/lib64 ;;
   esac
}

cuda-path-add () 
{ 
    local dir=$1;
    : only prepend the dir when not already there;
    [ "${PATH/$dir}" == "${PATH}" ] && export PATH=$dir:$PATH
}

cuda-libpath-add () 
{ 
    local dir=$1;
    : only prepend the dir when not already there;
    [ "${LD_LIBRARY_PATH/$dir}" == "${LD_LIBRARY_PATH}" ] && export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
}





cuda-path(){
    local dir=$(cuda-dir)
    [ ! -d $dir ] && return 1
    cuda-path-add $dir/bin

    local libdir=$(cuda-libdir)
    cuda-libpath-add $libdir
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

cuda-pdf-(){ echo $(cuda-dir)/doc/pdf/${1:-CUDA_C_Programming_Guide}.pdf ; }
cuda-pdf(){ open $(cuda-pdf-)  ; }
cuda-curand(){ open $(cuda-pdf- CURAND_Library) ; }



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



cuda-prefix-default(){ echo /usr/local/cuda ; }
cuda-prefix(){ echo ${OPTICKS_CUDA_PREFIX:-$(cuda-prefix-default)} ; }
cuda-libdir-(){ cat << EOD
$(cuda-prefix)/lib64
$(cuda-prefix)/lib
EOD
}
cuda-libdir(){
   local dir
   $FUNCNAME- | while read dir ; do 
     [ -d "$dir" ] && echo $dir
   done
}


cuda-pc-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/OpticksCUDA.pc ; }
cuda-pc-(){ 
  local prefix=$(cuda-prefix)
  local includedir=${prefix}/include
  local libdir=$(cuda-libdir)

  cat << EOP

## $FUNCNAME
## NB no prefix variable, as this prevents --define-prefix from having any effect 
## This is appropriate with CUDA as it is 
## a system install, not something that is distributed OR relocatable.   

includedir=$includedir
libdir=$libdir

Name: CUDA
Description: 
Version: 9.1 
Libs: -L\${libdir} -lcudart -lcurand
Cflags: -I\${includedir}

EOP
}


cuda-pc(){
   local msg="=== $FUNCNAME :"
   local path=$(cuda-pc-path)
   local dir=$(dirname $path)
   [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir 
   echo $msg $path 
   cuda-pc- > $path 
}


# cuda-setup is too crucial to hide here, moved to opticks-
cuda-setup(){ cat << EOS
# $FUNCNAME
EOS
}


