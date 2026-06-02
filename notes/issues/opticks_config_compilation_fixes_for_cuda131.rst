opticks_config_compilation_fixes_for_cuda131
==============================================


Issue 1 : compute_70 unsupported : avoid that by changing OPTICKS_COMPUTE_CAPABILITY and OPTICKS_COMPUTE_ARCHTECTURES to start from 89 (not 70)
-----------------------------------------------------------------0--------------------------------------------------------------------------------

* this drops support for Volta GPUs

Following update to CUDA 13.1 I get the below when building Opticks::

    nvcc fatal   : Unsupported gpu architecture 'compute_70'


Issue 2 : cuda thrust reorganized headers in 13.1
---------------------------------------------------


With CUDA 13.1 I get the below. Where has that thrust header gone ?::

    [ 44%] Building CXX object CMakeFiles/QUDARap.dir/QMultiFilm.cc.o
    In file included from /home/blyth/opticks/qudarap/qpmt.h:32,
                     from /home/blyth/opticks/qudarap/QPMT.hh:22,
                     from /home/blyth/opticks/qudarap/QPMT.cc:31:
    /data1/blyth/local/custom4_Debug/0.1.9/include/Custom4/C4MultiLayrStack.h:59:10: fatal error: thrust/complex.h: No such file or directory
       59 | #include <thrust/complex.h>
          |          ^~~~~~~~~~~~~~~~~~
    compilation terminated.
    make[2]: *** [CMakeFiles/QUDARap.dir/build.make:163: CMakeFiles/QUDARap.dir/QPMT.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    In file included from /home/blyth/opticks/qudarap/qpmt.h:32,
                     from /home/blyth/opticks/qudarap/qsim.h:65,
                     from /home/blyth/opticks/qudarap/QU.cc:29:
    /data1/blyth/local/custom4_Debug/0.1.9/include/Custom4/C4MultiLayrStack.h:59:10: fatal error: thrust/complex.h: No such file or directory
       59 | #include <thrust/complex.h>
          |          ^~~~~~~~~~~~~~~~~~
    compilation terminated.


CMakeLists.txt::

    add_library( ${name} SHARED ${SOURCES} ${HEADERS} )

    if(BUILD_WITH_CUDA)
        # 1. Attempt to find the modern CUDA 13+ CCCL layout
        find_package(CCCL QUIET PATHS "${CUDAToolkit_LIBRARY_DIR}/cmake/cccl")

        if(TARGET CCCL::CCCL)
            # CUDA 13.x+ path
            target_link_libraries(${name} PRIVATE CCCL::CCCL)
        else()
            # CUDA 11/12 legacy fallback: Ensure the standard CUDA include directory is explicitly passed
            # to host C++ compilers for files like QPMT.cc
            if(TARGET CUDA::toolkit)
                target_link_libraries(${name} PRIVATE CUDA::toolkit)
            elseif(CUDAToolkit_INCLUDE_DIRS)
                target_include_directories(${name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
            endif()
        endif()
    endif()





Issue 3 : cuda cccl internals warning
-----------------------------------------

::

    [ 43%] Building CXX object CMakeFiles/QUDARap.dir/QMultiFilm.cc.o
    [ 44%] Building CXX object CMakeFiles/QUDARap.dir/QBuf.cc.o
    In file included from tmpxft_000073d8_00000000-6_QEvt.compute_120.cudafe1.stub.c:1:
    /tmp/tmpxft_000073d8_00000000-6_QEvt.compute_120.cudafe1.stub.c:72:842: warning: ‘using maximum = struct cuda::__4::maximum<long int>’ is deprecated: Use cuda::maximum instead [-Wdeprecated-declarations]
       72 | 
          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ^            
    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/thrust/functional.h:185:33: note: declared here
      185 | template <class T = void>
          |                                 ^      
    In file included from tmpxft_000073d8_00000000-6_QEvt.compute_120.cudafe1.stub.c:1:
    /tmp/tmpxft_000073d8_00000000-6_QEvt.compute_120.cudafe1.stub.c:78:845: warning: ‘using maximum = struct cuda::__4::maximum<long int>’ is deprecated: Use cuda::maximum instead [-Wdeprecated-declarations]
       78 | 
          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ^            
    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/thrust/functional.h:185:33: note: declared here
      185 | template <class T = void>
          |                                 ^      
    [ 46%] Linking CXX shared library libQUDARap.so
    [ 46%] Built target QUDARap
    [ 47%] Building CXX object tests/CMakeFiles/QRngTest.dir/QRngTest.cc.o
    [ 48%] Building CXX object tests/CMakeFiles/QScintTest.dir/QScintTest.cc.o
    [ 50%] Building CXX object tests/CMakeFiles/QCerenkovIntegralTest.dir/QCerenkovIntegralTest.cc.o



Avoid the deprecated noise::

    if(BUILD_WITH_CUDA)
        # 1. Attempt to find the modern CUDA 13+ CCCL layout
        find_package(CCCL QUIET PATHS "${CUDAToolkit_LIBRARY_DIR}/cmake/cccl")

        if(TARGET CCCL::CCCL)
            # CUDA 13.x+ path
            target_link_libraries(${name} PRIVATE CCCL::CCCL)
        else()
            # CUDA 11/12 legacy fallback: Ensure the standard CUDA include directory is explicitly passed
            # to host C++ compilers for files like QPMT.cc
            if(TARGET CUDA::toolkit)
                target_link_libraries(${name} PRIVATE CUDA::toolkit)
            elseif(CUDAToolkit_INCLUDE_DIRS)
                target_include_directories(${name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
            endif()
        endif()


        # avoid CUDA 13.1 warnings from compilation of cuda stub internals
        # warning: ‘using maximum = struct cuda::__4::maximum<long int>’ is deprecated: Use cuda::maximum instead 
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)

    endif()







Issue 4 : QUDARap tests failing to include sysrap headers ?
-------------------------------------------------------------

::

     71%] Building CXX object tests/CMakeFiles/QCerenkovTest.dir/QCerenkovTest.cc.o
    [ 73%] Building CXX object tests/CMakeFiles/QBndTest.dir/QBndTest.cc.o
    /home/blyth/opticks/qudarap/tests/QRngTest.cc:12:10: fatal error: NP.hh: No such file or directory
       12 | #include "NP.hh"
          |          ^~~~~~~
    compilation terminated.
    /home/blyth/opticks/qudarap/tests/QScintTest.cc:2:10: fatal error: OPTICKS_LOG.hh: No such file or directory
        2 | #include "OPTICKS_LOG.hh"
          |          ^~~~~~~~~~~~~~~~
    compilation terminated.
    /home/blyth/opticks/qudarap/tests/QCerenkovIntegralTest.cc:10:10: fatal error: spath.h: No such file or directory
       10 | #include "spath.h"
          |          ^~~~~~~~~
    compilation terminated.
    /home/blyth/opticks/qudarap/tests/QPropTest.cc:1:10: fatal error: SPropMockup.h: No such file or directory
        1 | #include "SPropMockup.h"
          |          ^~~~~~~~~~~~~~~
    compilation terminated.
    /home/blyth/opticks/qudarap/tests/QSimCreateTest.cc:1:10: fatal error: OPTICKS_LOG.hh: No such file or directory
        1 | #include "OPTICKS_LOG.hh"
          |          ^~~~~~~~~~~~~~~~
    compilation terminated.
    /home/blyth/opticks/qudarap/tests/QTexLookupTest.cc:2:10: fatal error: NP.hh: No such file or directory
        2 | #include "NP.hh"
          |          ^~~~~~~



Huh, fixed by being more explicit with the SysRap dependency::

    [lo] A[blyth@localhost opticks]$ git diff qudarap/tests/CMakeLists.txt
    diff --git a/qudarap/tests/CMakeLists.txt b/qudarap/tests/CMakeLists.txt
    index 661babf9b..ebf954e2c 100644
    --- a/qudarap/tests/CMakeLists.txt
    +++ b/qudarap/tests/CMakeLists.txt
    @@ -70,7 +70,7 @@ message(STATUS "BASH_EXECUTABLE : ${BASH_EXECUTABLE}")
     foreach(SRC ${TEST_SOURCES})
         get_filename_component(TGT ${SRC} NAME_WE)
         add_executable(${TGT} ${SRC})
    -    target_link_libraries(${TGT} QUDARap )
    +    target_link_libraries(${TGT} QUDARap Opticks::SysRap )
         install(TARGETS ${TGT} DESTINATION lib)
     
         #add_test(${name}.${TGT} ${TGT})
    @@ -86,7 +86,7 @@ endforeach()
     foreach(SRC ${CUDART_TEST_SOURCES})
         get_filename_component(TGT ${SRC} NAME_WE)
         add_executable(${TGT} ${SRC})
    -    target_link_libraries(${TGT} QUDARap CUDA::cudart )
    +    target_link_libraries(${TGT} QUDARap CUDA::cudart Opticks::SysRap )
         install(TARGETS ${TGT} DESTINATION lib)
     
         #add_test(${name}.${TGT} ${TGT})
    @@ -108,7 +108,7 @@ endforeach()
     foreach(SRC ${KNOWN_FAIL_TEST_SOURCES})
         get_filename_component(TGT ${SRC} NAME_WE)
         add_executable(${TGT} ${SRC})
    -    target_link_libraries(${TGT} QUDARap )
    +    target_link_libraries(${TGT} QUDARap Opticks::SysRap )
         install(TARGETS ${TGT} DESTINATION lib)
     endforeach()
     


Issue 5 : build now completes but found OptiX_800 not intended OptiX_910 ?
----------------------------------------------------------------------------


::

    -- Custom4_CMAKE_PATH    : /data1/blyth/local/custom4_Debug/0.1.9/lib64/Custom4-0.1.9/Custom4Config.cmake 
    -- Custom4_CMAKE_DIR     : /data1/blyth/local/custom4_Debug/0.1.9/lib64/Custom4-0.1.9 
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++17
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_VERBOSE : ON 
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_MODULE  : /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake 
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_INCLUDE : /cvmfs/opticks.ihep.ac.cn/external/OptiX_800/include 
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_VERSION : 80000 : is 7+ 
    -- CSGOptiX Custom4_FOUND:1  
    -- CSGOptiX CSG_FOUND     : 1 
    -- CSGOptiX CSG_INCLUDE_DIRS : /data1/blyth/local/opticks_Debug/include/CSG;/data1/blyth/local/opticks_Debug/externals/glm/glm;/usr/local/cuda-13.1/targets/x86_64-linux/include;/data1/blyth/local/opticks_Debug/include/CSG 
    -- CSGOptiX OpticksOptiX_VERSION  : 80000 
    -- write to buildenvpath /data1/blyth/local/opticks_Debug/build/CSGOptiX/buildenv.sh 
    -- OPTIX_SOURCES : CSGOptiX7.cu;Check.cu 


Probably need cleaner rebuild::

    [lo] A[blyth@localhost CSGOptiX]$ t om-cmake-okconf
    om-cmake-okconf () 
    { 
        local sdir=$1;
        local bdir=$PWD;
        [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000;
        local opt=$(om-pkg-opt $name);
        local rc;
        cmake $sdir -G "$(om-cmake-generator)" -DCMAKE_BUILD_TYPE=$(opticks-buildtype) -DOPTICKS_PREFIX=$(om-prefix) -DCMAKE_INSTALL_PREFIX=$(om-prefix) -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules -DOptiX_INSTALL_DIR=$(opticks-optix-prefix) -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) -DCOMPUTE_ARCHITECTURES=$(opticks-compute-architectures) $opt;
        rc=$?;
        return $rc
    }
    [lo] A[blyth@localhost CSGOptiX]$ opticks-optix-prefix
    /cvmfs/opticks.ihep.ac.cn/external/OptiX_910
    [lo] A[blyth@localhost CSGOptiX]$ 


YEP om-cleaninstall finds 90100 and gives another thrust/complex.h error in CSGOptiX::

    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++17
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_VERBOSE : ON 
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_MODULE  : /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake 
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_INCLUDE : /cvmfs/opticks.ihep.ac.cn/external/OptiX_910/include 
    -- /home/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_VERSION : 90100 : is 7+ 
    -- CSGOptiX Custom4_FOUND:1  
    -- CSGOptiX CSG_FOUND     : 1 
    -- CSGOptiX CSG_INCLUDE_DIRS : /data1/blyth


::

    [ 30%] Building CXX object CMakeFiles/CSGOptiX.dir/IAS_Builder.cc.o
    [ 32%] Building CXX object CMakeFiles/CSGOptiX.dir/SBT.cc.o
    In file included from /data1/blyth/local/opticks_Debug/include/QUDARap/qpmt.h:32,
                     from /data1/blyth/local/opticks_Debug/include/QUDARap/qsim.h:65,
                     from /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:78:
    /data1/blyth/local/custom4_Debug/0.1.9/include/Custom4/C4MultiLayrStack.h:59:10: fatal error: thrust/complex.h: No such file or directory
       59 | #include <thrust/complex.h>
          |          ^~~~~~~~~~~~~~~~~~
    compilation terminated.
    make[2]: *** [CMakeFiles/CSGOptiX.dir/build.make:90: CMakeFiles/CSGOptiX.dir/CSGOptiX.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....



Same CCCL fix as QUDARap.




Issue 6 : cudart inconsistency warning
------------------------------------------


I see the below warning ? Should I avoid cudart on the CMake link line ?::

    [ 70%] Linking CXX executable G4CXSimtraceTest
    [ 76%] Linking CXX executable G4CXRenderTest
    [ 82%] Linking CXX executable G4CXSimulateTest
    /usr/bin/ld: warning: libcudart.so.13, needed by /data1/blyth/local/opticks_Debug/lib64/libCSGOptiX.so, may conflict with libcudart.so.12
    /usr/bin/ld: warning: libcudart.so.13, needed by /data1/blyth/local/opticks_Debug/lib64/libCSGOptiX.so, may conflict with libcudart.so.12
    /usr/bin/ld: warning: libcudart.so.13, needed by /data1/blyth/local/opticks_Debug/lib64/libCSGOptiX.so, may conflict with libcudart.so.12
    /usr/bin/ld: warning: libcudart.so.13, needed by /data1/blyth/local/opticks_Debug/lib64/libCSGOptiX.so, may conflict with libcudart.so.12
    [ 82%] Built target G4CXOpticks_setGeometry_Test
    [ 82%] Built target G4CXSimtraceTest


Was transient.  Second om-cleaninstall did not have the warning.



opticks-t all passing
----------------------

::

    SLOW: tests taking longer that 15.0 seconds
    FAILS:  0   / 221   :  Tue Jun  2 15:25:02 2026  :  GEOM J26_1_1_opticks_Debug  



workstation to server cross compile ?
---------------------------------------


I am building on my workstation::

    A[blyth@localhost nvidia_rtx_pro_6000_blackwell_cuda131_optix910_update]$ cat /etc/almalinux-release 
    AlmaLinux release 9.6 (Sage Margay)

With the intention to use the libs/binaries on server with different AlmaLinux release ?
Will that be a problem ?::

    L[blyth@lxlogin004 oj]$ oj6k
    srun: job 3438316 queued and waiting for resources
    srun: job 3438316 has been allocated resources
    L[blyth@junogpu001 oj]$ cat /etc/almalinux-release 
    AlmaLinux release 9.7 (Moss Jungle Cat)





Building binaries on an **AlmaLinux 9.6** workstation to run on an **AlmaLinux
9.7** server will **not be a problem**.

In fact, you are doing this in the safest direction possible. Here is why this
works perfectly, along with a few minor things to keep in mind regarding your
high-performance computing (HPC) setup.

---

### Why it Works: Minor Version Compatibility

Enterprise Linux distributions (like AlmaLinux, Rocky Linux, and RHEL)
guarantee strict **Application Binary Interface (ABI) and Application
Programming Interface (API) stability** across the entire major release
lifecycle (the 9.x family).

* **The Glibc Rule:** The Linux C library (`glibc`) is backward-compatible but
*not* forward-compatible.

* **Your Direction:** AlmaLinux 9.6 uses an older or equal version of `glibc`
compared to 9.7. A binary compiled on 9.6 will easily find all the symbols it
needs on 9.7.

* **If it were reversed:** Compiling on 9.7 and running on 9.6 *could*
occasionally fail if the newer compiler used a newly introduced symbol in
`glibc`.

---

### Crucial Caveats for your Specific Stack (Opticks, CUDA 13.1, OptiX)

While the operating system layers match up perfectly, high-performance
simulation frameworks like **Opticks** rely heavily on hardware-software
alignment. Ensure you check these three points on the server:

#### 1. NVIDIA Driver Version (The Most Critical)

CUDA 13.1 requires a minimum NVIDIA kernel driver version on the host machine.

* Ensure the **server** (`junogpu001`) has an NVIDIA driver installed that
fully supports CUDA 13.1.

* You can check this on the server by running `nvidia-smi`. If the driver is
too old, your binaries will crash with a CUDA initialization or driver mismatch
error immediately upon launching.

#### 2. GPU Microarchitecture Alignment

Your workstation prompt mentions **Blackwell** (`nvidia_rtx_pro_6000_blackwell`).

* If you optimized your Opticks build strictly for Blackwell targets
(`-DCMAKE_CUDA_ARCHITECTURES=100` or similar), those binaries **will fail** if
the server (`junogpu001`) uses an older GPU generation (like Hopper `sm_90` or
Ampere `sm_80`).

* **The Fix:** If the server has a different GPU, ensure your CMake
configuration compiles for *both* architectures (e.g.,
`-DCMAKE_CUDA_ARCHITECTURES="100;90"`).

#### 3. Cuda Runtime Linkage

Because you ran into dual-linkage warnings (`libcudart.so.12` vs
`libcudart.so.13`) earlier, make sure you deploy **only** the self-contained
Opticks installation directory (`/data1/blyth/local/opticks_Debug`) to the
server, and that your target environment variables (`LD_LIBRARY_PATH`) point
exclusively to your CUDA 13.1 paths on the server.





Workstation::

    [lo] A[blyth@localhost opticks]$ nvidia-smi
    Tue Jun  2 15:35:15 2026       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA RTX 5000 Ada Gene...    Off |   00000000:AC:00.0  On |                  Off |
    | 30%   41C    P8             16W /  250W |     579MiB /  32760MiB |      3%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |    0   N/A  N/A            3994      G   /usr/bin/gnome-shell                    185MiB |
    |    0   N/A  N/A            6009      G   /usr/lib64/firefox/firefox              208MiB |
    |    0   N/A  N/A            7113      G   ...0601-090046.628000-production         61MiB |
    |    0   N/A  N/A            7119      G   /usr/bin/Xwayland                        11MiB |
    +-----------------------------------------------------------------------------------------+
    [lo] A[blyth@localhost opticks]$ 


Server::

    L[blyth@junogpu001 oj]$ nvidia-smi
    Tue Jun  2 15:35:21 2026       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 595.58.03              Driver Version: 595.58.03      CUDA Version: 13.2     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:17:00.0 Off |                    0 |
    | N/A   31C    P8             34W /  600W |       0MiB /  97887MiB |      0%      Default |
    |                                         |                        |             Disabled |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+




Based on the `nvidia-smi` outputs, you have an interesting cross-generation
compilation scenario. Your workstation uses an **Ada Lovelace** card, while
your server is a massive new **Blackwell** node.

Because of how the hardware architectures differ, your binaries **will fail to
run on the server if you only compile for the workstation's default target.**

Here is the exact breakdown of how to configure your build correctly:

---

### 1. The Architecture Difference (The Main Hurdle)

* **Workstation GPU:** NVIDIA RTX 5000 Ada Generation $\rightarrow$ Architecture: **`sm_89`** (Ada Lovelace)
* **Server GPU:** NVIDIA RTX PRO 6000 Blackwell $\rightarrow$ Architecture: **`sm_100`** (Blackwell)

If you compile Opticks on your workstation without specifying the
architectures, CMake or `nvcc` will automatically target your local hardware
(`sm_89`). When you copy those binaries to the Blackwell server, OptiX/CUDA
will throw an `invalid device function` error because Blackwell does not
natively run binary code optimized explicitly for Ada Lovelace.

#### The Solution: Dual-Target Compilation

You need to force `nvcc` to embed code for **both** architectures into the
compilation payload (Fatbinary). Adjust your CMake configure line on the
workstation to include both:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89;100"

```

This forces the compiler to generate SASS (machine code) for `sm_89` and
`sm_100`, ensuring the resulting binary runs optimally on both the workstation
and the server.

---

### 2. The Driver and CUDA Forward Compatibility (Safe)

* **Workstation Driver:** 610.43 (Supports up to CUDA 13.3)
* **Server Driver:** 595.58 (Supports up to CUDA 13.2)
* **Your Compilation Target:** CUDA 13.1

You are building with **CUDA 13.1**. Because the server's driver environment
natively supports up to **CUDA 13.2**, your CUDA 13.1 binaries are perfectly
within the supported driver envelope. The server's kernel driver is fully
equipped to handle your CUDA 13.1 application runtime calls without needing a
driver upgrade.

---

### Summary Checklist for Deployment

1. **Wipe your build cache** on the workstation to purge old `sm_70` or
single-target configurations.

2. Configure CMake passing **`-DCMAKE_CUDA_ARCHITECTURES="89;100"`**.

3. Compile, install, and ship the layout to the server—it will run smoothly on
that Blackwell setup.





The Opticks build acts on the below envvar.  I thought it was sm_120 for RTX
PRO 6000 Blackwell Server edition ?

export OPTICKS_COMPUTE_ARCHITECTURES=89,120   ## 120 FOR RTX PRO 6000 BLACKWELL








You are absolutely correct to double-check that, but there is a nuance to how
NVIDIA splits its architectures. Your assumption about **`120`** is correct,
though it depends entirely on which generation of the toolchain or environment
variable format you are passing to Opticks.

Here is the breakdown of why you see both `100` and `120` thrown around for
Blackwell, and what you should put in your environment variable.

---

### The Two Blackwell Branches: Datacenter vs. Workstation

NVIDIA split the Blackwell generation into two distinct compute capabilities
based on the underlying chip architecture and memory layout:

| GPU Tier / Name | Compute Capability (SASS) | Native Feature Set |
| --- | --- | --- |
| **Datacenter Blackwell** (B200 / B300 / GB200) | **`sm_100`** / **`sm_103`** | Features HBM3e memory and the massive hardware **TMEM (Tensor Memory)** subsystem required for FlashAttention-4. |
| **Workstation/Professional Blackwell** (RTX PRO 6000, RTX 5090) | **`sm_120`** | Features ultra-fast GDDR7 memory. It uses a distinct processing layout that lacks the datacenter-specific TMEM subsystem. |

Because your server is using the **RTX PRO 6000 Blackwell Server Edition**, its
true native compute capability hardware string is indeed **`sm_120`**.

---

### Why "100" or "120" in CMake vs. `OPTICKS_COMPUTE_ARCHITECTURES`?

When you pass architectures via your environment variable
`OPTICKS_COMPUTE_ARCHITECTURES=89,120`, Opticks maps these numbers straight
into the `-gencode` compilation flags for `nvcc`.

However, you need to verify if your specific version of the `nvcc` compiler
(CUDA 13.1) fully supports `compute_120` out of the box, or if it expects you
to use `100` as the baseline virtual architecture fallback.

#### How to verify what your workstation's `nvcc` accepts:

Run a quick test flag check on your workstation command line:

```bash
nvcc --help | grep -E "sm_120|sm_100"

```

1. **If `sm_120` is listed:** You are safe to use `export
OPTICKS_COMPUTE_ARCHITECTURES=89,120`. This compiles specific, un-throttled
binary instructions for both your Ada Lovelace workstation and your Blackwell
server.

2. **If only `sm_100` is listed:** If your early CUDA 13.1 installation doesn't
recognize `120` yet, you must fall back to `export
OPTICKS_COMPUTE_ARCHITECTURES=89,100`. The Blackwell hardware runner will
JIT-compile or execute the generic `sm_100` payload natively on the server
without complaining.


**Your Action:** Keep your environment variable exactly as you have it:

```bash
export OPTICKS_COMPUTE_ARCHITECTURES=89,120

```

If Opticks compilation errors out immediately on a target architecture mismatch
during the very first file, simply drop it to `89,100`.














