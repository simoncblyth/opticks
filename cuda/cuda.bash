# === func-gen- : cuda/cuda fgp cuda/cuda.bash fgn cuda fgh cuda
cuda-src(){      echo cuda/cuda.bash ; }
cuda-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cuda-src)} ; }
cuda-vi(){       vi $(cuda-source) ; }
cuda-env(){      elocal- ; cuda-path ; }
cuda-usage(){ cat << EOU

CUDA
======

tips to free up GPU memory
---------------------------

#. minimise the number of apps and windows running
#. put the machine to sleep and go and have a coffee



CUDA Driver and Runtime API interop
------------------------------------

* http://stackoverflow.com/questions/20539349/cuda-runtime-api-and-dynamic-kernel-definition

  * http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER


Bit packing with CUDA vector types
-------------------------------------

* http://nvlabs.github.io/cub/index.html




Updating to CUDA 6.5 (Feb 2, 2015)
----------------------------------

Using sysprefs panel to initiate the install, 
going from:

* CUDA Driver Version: 5.5.47
* GPU Driver Version: 8.26.26 310.40.45f01

To:

* (No newer CUDA driver available)
* CUDA Driver Version: 6.5.45   
* GPU Driver Version: 8.26.26 310.40.45f01


version available
------------------

From system prefs::

    Available: CUDA 6.5.18 Driver update is available


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




samples install
------------------

::

    cuda-samples-install
    cuda-samples-cd

    make




versions
---------

::

   Current: CUDA Driver Version: 5.5.47
             GPU Driver Version: 8.26.26 310.40.45f01

    delta:~ blyth$ cuda-
    delta:~ blyth$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2013 NVIDIA Corporation
    Built on Thu_Sep__5_10:17:14_PDT_2013
    Cuda compilation tools, release 5.5, V5.5.0
    delta:~ blyth$ 


installer pkginfo
~~~~~~~~~~~~~~~~~~

::

    installer -pkginfo -pkg cuda-mac-5.5.28_10.9_64.pkg 
    CUDA 5.5
    CUDA Driver
    CUDA Toolkit
    CUDA Samples


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


CUDA CMAKE Mavericks
-----------------------

Informative explanation of Mavericks CUDA challenges wrt compiler flags...
 
https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks

See::

   cudawrap-
   thrustrap-
   thrusthello-
   thrust-



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
cuda-tmp(){ echo $(local-tmp)/env/cuda ; }
cuda-cd(){  cd $(cuda-dir); }
cuda-mate(){ mate $(cuda-dir) ; }
cuda-writable-dir(){ echo $(local-base)/env/cuda ; } 


cuda-nvcc-flags(){
    case $NODE_TAG in 
       D) echo -ccbin /usr/bin/clang --use_fast_math ;;
       *) echo --use_fast_math ;; 
    esac 
}

cuda-export()
{
   echo -n
}


cuda-idir(){ echo $(cuda-dir)/include ; }
cuda-icd(){  cd $(cuda-idir); }
cuda-find(){ find $(cuda-idir) -name '*.h' -exec grep -H ${1:-cudaGraphics} {} \; ; }

cuda-dir(){ 
   case $NODE_TAG in 
     D) echo /Developer/NVIDIA/CUDA-5.5 ;;
     G1) echo /usr/local/cuda-5.5  ;;
   esac
}

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

    if [ "$NODE_TAG" == "D" ]; then 
       export DYLD_LIBRARY_PATH=$libdir:$DYLD_LIBRARY_PATH      # not documented ??? links from this to $dir/lib
    else
       export LD_LIBRARY_PATH=$libdir:$LD_LIBRARY_PATH
    fi

    # these are not seen by the pycuda build
}

cuda-url(){
   case $(uname) in 
     Darwin) echo http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/cuda-mac-5.5.28_10.9_64.pkg ;;
   esac
}
cuda-pkg(){
   echo $(basename $(cuda-url))
}

cuda-get(){
   local dir=$(dirname $(cuda-tmp)) &&  mkdir -p $dir && cd $dir
   local url=$(cuda-url)
   local pkg=$(cuda-pkg)
   [ -d $(cuda-dir) ] && echo $msg CUDA is already installed at $(cuda-dir) && return 0 
   [ ! -f "$pkg" ] && curl -L -O $url 
   open $pkg   # GUI installer
}

cuda-osx-pkginfo(){
   installer -pkginfo -pkg $(dirname $(cuda-tmp))/$(cuda-pkg)
}

cuda-osx-getting-started(){
   open $(cuda-dir)/doc/html/cuda-getting-started-guide-for-mac-os-x/index.html
}

cuda-guide(){
   open $(cuda-dir)/doc/html/cuda-c-programming-guide/index.html
}

cuda-doc(){
   open $(cuda-dir)/doc/html/index.html
}

cuda-osx-kextstat(){
   kextstat | grep -i cuda
}

cuda-samples-dir(){ echo $(cuda-writable-dir)/NVIDIA_CUDA-5.5_Samples ; }
cuda-samples-cd(){ cd $(cuda-samples-dir) ; }
cuda-samples-make(){
   cuda-samples-cd
   make $*
}
cuda-samples-install(){
   local iwd=$PWD
   local dir=$(cuda-writable-dir)
   [ ! -d "$dir" ] && mkdir -p $dir
  
   # cuda-install-samples-5.5.sh $dir
   # have to do it manually to skip the doc folder which goes over afs quota

   local src=$(dirname $(dirname $(which cuda-install-samples-5.5.sh)))/samples
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
