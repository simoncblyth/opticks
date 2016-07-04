# === func-gen- : cuda/cudatoolkit fgp cuda/cudatoolkit.bash fgn cudatoolkit fgh cuda
cudatoolkit-src(){      echo cuda/cudatoolkit.bash ; }
cudatoolkit-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cudatoolkit-src)} ; }
cudatoolkit-vi(){       vi $(cudatoolkit-source) ; }
cudatoolkit-env(){      olocal- ; }
cudatoolkit-usage(){ cat << EOU

CUDA TOOLKIT
==============

* https://developer.nvidia.com/cuda-toolkit
* https://developer.nvidia.com/cuda-downloads
* http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html

GPU Hardware Check
--------------------

::

    /sbin/lspci | grep -i nvidia


Versions
---------

* https://developer.nvidia.com/cuda-toolkit-archive

CUDA Toolkit 5.5 (July 2013)
CUDA Toolkit 5.0 (Oct 2012)
CUDA Toolkit 4.2 (April 2012)
CUDA Toolkit 4.1 (Jan 2012)
CUDA Toolkit 4.0 (May 2011)

4.1
----

http://chroma.bitbucket.org/install/details.html mentions CUDA TOOLKIT 4.1
https://developer.nvidia.com/cuda-toolkit-41-archive

N belle7  (SL 5.1 Boron)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/cudatoolkit_4.1.28_linux_32_rhel5.x.run

GPU on this machine is not capable, but we can try ocelot- emulation first::

    [blyth@belle7 cuda]$ sudo sh cudatoolkit_4.1.28_linux_32_rhel5.x.run
    Verifying archive integrity... All good.
    Uncompressing NVIDIA CUDA.....
    ........................................................................
    Enter install path (default /usr/local/cuda, '/cuda' will be appended): 
    `include' -> `/usr/local/cuda/include'
    `include/texture_fetch_functions.h' -> `/usr/local/cuda/include/texture_fetch_functions.h'
    `include/driver_functions.h' -> `/usr/local/cuda/include/driver_functions.h'
    ...
    ========================================

    * Please make sure your PATH includes /usr/local/cuda/bin
    * Please make sure your LD_LIBRARY_PATH
    *   for 32-bit Linux distributions includes /usr/local/cuda/lib
    *   for 64-bit Linux distributions includes /usr/local/cuda/lib64:/usr/local/cuda/lib
    * OR
    *   for 32-bit Linux distributions add /usr/local/cuda/lib
    *   for 64-bit Linux distributions add /usr/local/cuda/lib64 and /usr/local/cuda/lib
    * to /etc/ld.so.conf and run ldconfig as root

    * Please read the release notes in /usr/local/cuda/doc/

    * To uninstall CUDA, delete /usr/local/cuda
    * Installation Complete

::

    [blyth@belle7 dyb]$ cudatoolkit-
    [blyth@belle7 dyb]$ cudatoolkit-path
    [blyth@belle7 dyb]$ which nvcc
    /usr/local/cuda/bin/nvcc
    [blyth@belle7 dyb]$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2011 NVIDIA Corporation
    Built on Thu_Jan_12_14:36:13_PST_2012
    Cuda compilation tools, release 4.1, V0.2.1221




EOU
}
cudatoolkit-dir(){ echo $(local-base)/env/cuda/cuda-cudatoolkit ; }
cudatoolkit-cd(){  cd $(cudatoolkit-dir); }
cudatoolkit-url(){ echo http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/cudatoolkit_4.1.28_linux_32_rhel5.x.run ; }
#cudatoolkit-url(){ echo  http://developer.download.nvidia.com/compute/cuda/4_0/sdk/gpucomputingsdk_4.0.17_linux.run ; }
cudatoolkit-name(){ echo $(basename $(cudatoolkit-url)) ; }

cudatoolkit-path(){
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
   export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
}

cudatoolkit-mate(){ mate $(cudatoolkit-dir) ; }
cudatoolkit-get(){
   local dir=$(dirname $(cudatoolkit-dir)) &&  mkdir -p $dir && cd $dir
   
   local url=$(cudatoolkit-url)
   local name=$(cudatoolkit-name)
   [ ! -f "$name" ] && curl -L -O $url

}
