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


version available
------------------

From system prefs::

    Available: CUDA 6.5.18 Driver update is available


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




FUNCTIONS
---------


cuda-get
       




EOU
}
cuda-tmp(){ echo $(local-tmp)/env/cuda ; }
cuda-cd(){  cd $(cuda-dir); }
cuda-mate(){ mate $(cuda-dir) ; }
cuda-writable-dir(){ echo $(local-base)/env/cuda ; } 
cuda-dir(){ echo /Developer/NVIDIA/CUDA-5.5 ; }
cuda-path(){
    local dir=$(cuda-dir)
    [ ! -d $dir ] && return 1
    export PATH=$dir/bin:$PATH
    #export DYLD_LIBRARY_PATH=$dir/lib:$DYLD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH      # not documented ??? links from this to $dir/lib

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
cuda-samples-install(){
   local dir=$(cuda-writable-dir)
   [ ! -d "$dir" ] && mkdir -p $dir
   cuda-install-samples-5.5.sh $dir
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



