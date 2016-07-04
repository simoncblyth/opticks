# === func-gen- : cuda/cudainstall fgp cuda/cudainstall.bash fgn cudainstall fgh cuda
cudainstall-src(){      echo cuda/cudainstall.bash ; }
cudainstall-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cudainstall-src)} ; }
cudainstall-vi(){       vi $(cudainstall-source) ; }
cudainstall-env(){      elocal- ; }
cudainstall-usage(){ cat << EOU

CUDA Install
=============

See Also
---------

* cuda-
* cudatoolkit-

Refs
-----

* https://developer.nvidia.com/cuda-downloads

Many Versions to Get Straight (status June 29, 2015)
-------------------------------------------------------

SysPrefs CUDA panel
~~~~~~~~~~~~~~~~~~~~

Before 7.0 install::

    CUDA Driver Version: 6.5.45
    GPU Driver Version: 8.26.26 310.40.45f01

    CUDA 7.0.36 Driver is available


After 7.0 install::

    CUDA Driver Version: 7.0.29
    GPU Driver Version: 8.26.26 310.40.45f01

    CUDA 7.0.36 Driver is available


deviceQuery : Driver Version / Runtime Version   6.5/5.5 -> 7.0/7.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 7.0 install::

    simon:cuda blyth$ cuda-deviceQuery 
    running /usr/local/env/cuda/NVIDIA_CUDA-5.5_Samples/bin/x86_64/darwin/release/deviceQuery
    /usr/local/env/cuda/NVIDIA_CUDA-5.5_Samples/bin/x86_64/darwin/release/deviceQuery Starting...

     CUDA Device Query (Runtime API) version (CUDART static linking)

    Detected 1 CUDA Capable device(s)

    Device 0: "GeForce GT 750M"
      CUDA Driver Version / Runtime Version          6.5 / 5.5
      CUDA Capability Major/Minor version number:    3.0
      Total amount of global memory:                 2048 MBytes (2147024896 bytes)
      ( 2) Multiprocessors, (192) CUDA Cores/MP:     384 CUDA Cores
      GPU Clock rate:                                926 MHz (0.93 GHz)


After 7.0 install and building samples::

    simon:~ blyth$ cuda-deviceQuery 
    running /usr/local/env/cuda/NVIDIA_CUDA-7.0_Samples/bin/x86_64/darwin/release/deviceQuery
    /usr/local/env/cuda/NVIDIA_CUDA-7.0_Samples/bin/x86_64/darwin/release/deviceQuery Starting...

     CUDA Device Query (Runtime API) version (CUDART static linking)

    Detected 1 CUDA Capable device(s)

    Device 0: "GeForce GT 750M"
      CUDA Driver Version / Runtime Version          7.0 / 7.0
      CUDA Capability Major/Minor version number:    3.0
      Total amount of global memory:                 2048 MBytes (2147024896 bytes)
      ( 2) Multiprocessors, (192) CUDA Cores/MP:     384 CUDA Cores
      GPU Max Clock rate:                            926 MHz (0.93 GHz)
      Memory Clock rate:                             2508 Mhz



nvcc version : Matches the Runtime Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 7.0 install::

    simon:cuda blyth$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2013 NVIDIA Corporation
    Built on Thu_Sep__5_10:17:14_PDT_2013
    Cuda compilation tools, release 5.5, V5.5.0
    simon:cuda blyth$ 

After 7.0::

    simon:cuda blyth$ which nvcc
    /Developer/NVIDIA/CUDA-7.0/bin/nvcc
    simon:cuda blyth$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2015 NVIDIA Corporation
    Built on Mon_Feb_16_23:23:36_CST_2015
    Cuda compilation tools, release 7.0, V7.0.27


What does the CUDA SysPref panel actually update ?
-----------------------------------------------------

Suspect a small subset of what the CUDA installer updates::

   /usr/local/cuda/lib/libcuda.dylib
   Frameworks/CUDA.framework 

Strings on prefPane binary support this::

    simon:~ blyth$ strings "/Library/PreferencePanes/CUDA Preferences.prefPane/Contents/MacOS/CUDA Preferences"  | more

    libcuda.dylib
    Frameworks/CUDA.framework
    /usr/local/cuda/lib
    /System/Library/Extensions/%@.kext
    ...
    /Library/LaunchAgents/com.nvidia.CUDASoftwareUpdate.plist


After CUDA 7.0 install::

    simon:release blyth$ otool -L /Library/Frameworks/CUDA.framework/Versions/Current/CUDA 
    /Library/Frameworks/CUDA.framework/Versions/Current/CUDA:
        @rpath/CUDA.framework/Versions/A/CUDA (compatibility version 1.1.0, current version 7.0.29)
        /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 855.17.0)
        /System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL (compatibility version 1.0.0, current version 1.0.0)
        /System/Library/Frameworks/IOKit.framework/Versions/A/IOKit (compatibility version 1.0.0, current version 275.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)


Getting Started Guide
-----------------------

* http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/

CUDA Driver

     /Library/Frameworks/CUDA.framework and the UNIX-compatibility stub /usr/local/cuda/lib/libcuda.dylib that refers to it.

CUDA Toolkit

     Toolkit supplements the CUDA Driver with compilers and additional libraries and header files that are installed 
     into /Developer/NVIDIA/CUDA-7.0 by default. Symlinks are created in /usr/local/cuda/ pointing 
     to their respective files in /Developer/NVIDIA/CUDA-7.0/. 
     Previous installations of the toolkit will be moved to /Developer/NVIDIA/CUDA-#.# 
     to better support side-by-side installations.

CUDA Samples (read-only)

      A read-only copy of the CUDA Samples is installed in /Developer/NVIDIA/CUDA-7.0/samples. 
      Previous installations of the samples will be moved to /Developer/NVIDIA/CUDA-#.#/samples 
      to better support side-by-side installations.


CUDA Mac Driver Check 06/29/2015
--------------------------------

* http://www.nvidia.com/object/mac-driver-archive.html

CUDA 7.0.36 driver for MAC   Release Date: 04/09/2015 << LATEST
CUDA 7.0.35 driver for MAC   Release Date: 04/02/2015
CUDA 7.0.29 driver for MAC   Release Date: 03/18/2015 << version installed with CUDA 7.0
CUDA 6.5.51 driver for MAC   Release Date: 04/21/2015
CUDA 6.5.46 driver for MAC   Release Date: 01/28/2015
CUDA 6.5.45 driver for MAC   Release Date: 01/28/2015 << CURRENT 
CUDA 6.5.37 driver for MAC   Release Date: 01/14/2015
CUDA 6.5.36 driver for MAC   Release Date: 01/14/2015
CUDA 6.5.33 driver for MAC   Release Date: 01/06/2015
CUDA 6.5.32 driver for MAC   Release Date: 12/19/2014
CUDA 6.5.25 driver for MAC   Release Date: 11/19/2014
CUDA 6.5.18 driver for MAC   Release Date: 09/19/2014
CUDA 6.5.14 driver for MAC   Release Date: 08/21/2014
CUDA 6.0.51 driver for MAC   Release Date: 07/03/2014
CUDA 6.0.46 driver for MAC   Release Date: 05/20/2014
CUDA 6.0.37 driver for MAC   Release Date: 04/16/2014
CUDA 5.5.47 driver for MAC   Release Date: 03/05/2014 


Annoying multiple versioning schemes for CUDA Drivers
-------------------------------------------------------

How does the CUDA R346 referred to from OptiX release notes
map to the Mac driver versions ? 

Note that /usr/local/cuda/lib/libcuda.dylib is not a symbolic link into the toolkit::

    simon:cuda blyth$ l /usr/local/cuda/lib/
    total 232
    -rwxr-xr-x  1 root  wheel  12372 Jan 15 15:14 libcuda.dylib
    lrwxr-xr-x  1 root  wheel     50 Jan 15  2014 libcublas.5.5.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcublas.5.5.dylib
    lrwxr-xr-x  1 root  wheel     46 Jan 15  2014 libcublas.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcublas.dylib
    lrwxr-xr-x  1 root  wheel     49 Jan 15  2014 libcublas_device.a -> /Developer/NVIDIA/CUDA-5.5/lib/libcublas_device.a
    lrwxr-xr-x  1 root  wheel     45 Jan 15  2014 libcudadevrt.a -> /Developer/NVIDIA/CUDA-5.5/lib/libcudadevrt.a
    lrwxr-xr-x  1 root  wheel     50 Jan 15  2014 libcudart.5.5.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcudart.5.5.dylib
    lrwxr-xr-x  1 root  wheel     46 Jan 15  2014 libcudart.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcudart.dylib

Version reported by otool matches mac driver archive. Before 7.0 install::

    simon:cuda blyth$ otool -L /usr/local/cuda/lib/libcuda.dylib
    /usr/local/cuda/lib/libcuda.dylib:
        /usr/local/cuda/lib/libcuda.dylib (compatibility version 1.1.0, current version 6.5.45)
        @rpath/CUDA.framework/Versions/A/CUDA (compatibility version 1.1.0, current version 6.5.45)
        /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 56.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 169.3.0)


After 7.0 install, *NB the change to libc++ rather than the old libstdc++*::

    simon:~ blyth$ otool -L /usr/local/cuda/lib/libcuda.dylib
    /usr/local/cuda/lib/libcuda.dylib:
        /usr/local/cuda/lib/libcuda.dylib (compatibility version 1.1.0, current version 7.0.29)
        @rpath/CUDA.framework/Versions/A/CUDA (compatibility version 1.1.0, current version 7.0.29)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)


CUDA RNNN drivers
--------------------

* :google:`CUDA R331 R343 R346 R355`

* http://nvidia.custhelp.com/app/answers/detail/a_id/3610/~/cve-2014-8298%3A-glx-indirect-(including-cve-2014-8093,-cve-2014-8098)



Optix 3.8.0 (May 2015) Release Notes 
--------------------------------------

The CUDA R346 or later driver is required. 
For the Mac, the driver extension module supplied with CUDA 7.0 will need to be installed.


CUDA Toolkit 4.0 – 7.0.

OptiX 3.8 has been built with CUDA 7.0, but any specified toolkit should work
when compiling PTX for OptiX. If an application links against both the OptiX
library and the CUDA runtime on Linux, it is recommended to use the same
version of CUDA as OptiX was built against.

gcc 4.4-4.8 have been tested on Linux. 

Xcode 5.1 has been tested on Mac OSX 10.9.





CUDA 7.0 Release note snippets
---------------------------------

* http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

On Mac OS X, libc++ is supported with XCode 5.x. 
Command-line option below is no longer needed when invoking NVCC::
 
      -Xcompiler -stdlib=libstdc++ 

Instead, NVCC uses the default library that Clang chooses on Mac OS X. 
Users are still able to choose between libc++ and libstdc++ 
by passing one of the below to NVCC.

     -Xcompiler -stdlib=libc++ 
     -Xcompiler -stdlib=libstdc++ 


OSX Layout
-----------

Aim to use the OptiX version layout (see optix-) with CUDA, 
ie versioned folders and a single top level symbolic link: 

* installer writes /Developer/NVIDIA/CUDA-5.5
* create CUDA symbolic link pointing at current version


Informative uninstall
----------------------

/usr/local/cuda/bin/uninstall::

     01 #!/usr/bin/perl
      2 
      3 use strict;
      4 
      5 my $whoami = qx{whoami};
      6 $whoami =~ /^root$/ or die "sudo required";
      7 
      8 system("kextunload /System/Library/Extensions/CUDA.kext");
      9 
     10 my @cuda_dirs = (
     11   "/usr/local/cuda",
     12   "/Developer/NVIDIA/CUDA-5.5",
     13   "/System/Library/StartupItems/CUDA",
     14   "/System/Library/Extensions/CUDA.kext",
     15   "/Library/Preferences/com.nvidia.CUDAPref.plist.lockfile",
     16   "/Library/Preferences/com.nvidia.CUDAPref.plist",
     17   "/Library/PreferencePanes/CUDA Preferences.prefPane",
     18   "/Library/LaunchAgents/com.nvidia.CUDASoftwareUpdate.plist",
     19   "/Library/Frameworks/CUDA.framework",
     20   "/private/var/db/receipts/com.nvidia.cuda.launchagent.pkg.plist",
     21 );
     22 chomp @cuda_dirs;
     23 
     24 foreach my $dir (@cuda_dirs) {
     25   print "removing [$dir]\n";
     26   system(qq{rm -rf "$dir"}) and die "could not remove [$dir]";
     27   -e "$dir" and die "$dir still exists";
     28 }



CUDA 5.5 Layout
-----------------

::

    simon:cuda blyth$ otool -L /Developer/NVIDIA/CUDA-5.5/lib/libcudart.5.5.dylib
    /Developer/NVIDIA/CUDA-5.5/lib/libcudart.5.5.dylib:
        @rpath/libcudart.5.5.dylib (compatibility version 0.0.0, current version 5.5.28)
        /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 635.21.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 159.1.0)
    simon:cuda blyth$ 


Hmm symbolic soup, maybe need to rename the below cuda to cuda-5.5 before install::

    simon:~ blyth$ l /usr/local/cuda/
    total 96
    drwxr-xr-x  28 root  wheel  952 Feb  2 12:46 lib
    lrwxr-xr-x   1 root  wheel   35 Jan 15  2014 EULA.txt -> /Developer/NVIDIA/CUDA-5.5/EULA.txt
    lrwxr-xr-x   1 root  wheel   30 Jan 15  2014 bin -> /Developer/NVIDIA/CUDA-5.5/bin
    lrwxr-xr-x   1 root  wheel   30 Jan 15  2014 doc -> /Developer/NVIDIA/CUDA-5.5/doc
    lrwxr-xr-x   1 root  wheel   33 Jan 15  2014 extras -> /Developer/NVIDIA/CUDA-5.5/extras
    lrwxr-xr-x   1 root  wheel   34 Jan 15  2014 include -> /Developer/NVIDIA/CUDA-5.5/include
    lrwxr-xr-x   1 root  wheel   36 Jan 15  2014 libnsight -> /Developer/NVIDIA/CUDA-5.5/libnsight
    lrwxr-xr-x   1 root  wheel   34 Jan 15  2014 libnvvp -> /Developer/NVIDIA/CUDA-5.5/libnvvp
    lrwxr-xr-x   1 root  wheel   31 Jan 15  2014 nvvm -> /Developer/NVIDIA/CUDA-5.5/nvvm
    lrwxr-xr-x   1 root  wheel   33 Jan 15  2014 open64 -> /Developer/NVIDIA/CUDA-5.5/open64
    lrwxr-xr-x   1 root  wheel   34 Jan 15  2014 samples -> /Developer/NVIDIA/CUDA-5.5/samples
    lrwxr-xr-x   1 root  wheel   30 Jan 15  2014 src -> /Developer/NVIDIA/CUDA-5.5/src
    lrwxr-xr-x   1 root  wheel   32 Jan 15  2014 tools -> /Developer/NVIDIA/CUDA-5.5/tools
    simon:~ blyth$ 

    simon:~ blyth$ l /usr/local/cuda/lib/
    total 232
    -rwxr-xr-x  1 root  wheel  12372 Jan 15 15:14 libcuda.dylib
    lrwxr-xr-x  1 root  wheel     50 Jan 15  2014 libcublas.5.5.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcublas.5.5.dylib
    lrwxr-xr-x  1 root  wheel     46 Jan 15  2014 libcublas.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcublas.dylib
    lrwxr-xr-x  1 root  wheel     49 Jan 15  2014 libcublas_device.a -> /Developer/NVIDIA/CUDA-5.5/lib/libcublas_device.a
    lrwxr-xr-x  1 root  wheel     45 Jan 15  2014 libcudadevrt.a -> /Developer/NVIDIA/CUDA-5.5/lib/libcudadevrt.a
    lrwxr-xr-x  1 root  wheel     50 Jan 15  2014 libcudart.5.5.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcudart.5.5.dylib
    lrwxr-xr-x  1 root  wheel     46 Jan 15  2014 libcudart.dylib -> /Developer/NVIDIA/CUDA-5.5/lib/libcudart.dylib



CUDA trawling
---------------

::

    /Library/LaunchAgents/com.nvidia.CUDASoftwareUpdate.plist

    /Library/Frameworks/CUDA.framework/Versions/A/Headers/cudaProfiler.h
    /Library/Frameworks/CUDA.framework/Versions/A/Headers/cudaGL.h
    /Library/Frameworks/CUDA.framework/Versions/A/Headers/cudadebugger.h
    /Library/Frameworks/CUDA.framework/Versions/A/Headers/cuda.h
    /Library/LaunchDaemons/com.nvidia.cuda.launcher.plist
    /System/Library/Extensions/CUDA.kext
    /Library/Frameworks/CUDA.framework
    /Library/Frameworks/CUDA.framework/Versions/A/CUDA



FindCUDA.cmake
---------------

::

    simon:~ blyth$ mdfind FindCUDA.cmake | grep -v Safari
    /Users/blyth/env/cuda/cuda.bash

    /usr/local/env/graphics/photonmap/CMake/FindCUDA.cmake
    /usr/local/env/graphics/hrt/cmake/Modules/FindCUDA.cmake
    /usr/local/env/optix/macrosim/macrosim_tracer/CMake/FindCUDA.cmake

    /opt/local/share/cmake-3.2/Modules/FindCUDA/run_nvcc.cmake
    /opt/local/share/cmake-3.2/Modules/FindCUDA/parse_cubin.cmake
    /opt/local/share/cmake-3.2/Modules/FindCUDA/make2cmake.cmake
    /opt/local/share/cmake-3.2/Modules/FindCUDA.cmake

    /usr/local/env/cuda/OptiX_370b2_sdk/CMake/FindCUDA.cmake
    /Developer/OptiX_370b2/SDK/CMake/FindCUDA/run_nvcc.cmake
    /Developer/OptiX_370b2/SDK/CMake/FindCUDA/parse_cubin.cmake
    /Developer/OptiX_370b2/SDK/CMake/FindCUDA/make2cmake.cmake
    /Developer/OptiX_370b2/SDK/CMake/FindCUDA.cmake
    /Developer/OptiX_370b2/SDK/CMakeLists.txt

    /Developer/OptiX_301/SDK/CMake/FindCUDA/run_nvcc.cmake
    /Developer/OptiX_301/SDK/CMake/FindCUDA/run_nvcc.cmake.original
    /Developer/OptiX_301/SDK/CMake/FindCUDA/parse_cubin.cmake
    /Developer/OptiX_301/SDK/CMake/FindCUDA/make2cmake.cmake
    /Developer/OptiX_301/SDK/CMake/FindCUDA.cmake
    /Developer/OptiX_301/SDK/CMakeLists.txt
    /Users/blyth/macports/cmake.log






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






EOU
}
cudainstall-dir(){ echo $(local-base)/env/cuda/cuda-cudainstall ; }
cudainstall-cd(){  cd $(cudainstall-dir); }
cudainstall-mate(){ mate $(cudainstall-dir) ; }
cudainstall-get(){
   local dir=$(dirname $(cudainstall-dir)) &&  mkdir -p $dir && cd $dir

}
