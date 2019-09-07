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

# === func-gen- : numerics/thrustrap/thrustrap fgp numerics/thrustrap/thrustrap.bash fgn thrustrap fgh numerics/thrustrap
thrap-src(){      echo thrustrap/thrustrap.bash ; }
thrap-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(thrap-src)} ; }
thrap-vi(){       vi $(thrap-source) ; }
thrap-usage(){ cat << EOU

ThrustRap
============



High Sierra 10.13.4 Xcode 9.2 (xcode-92) : latest plog warnings 
------------------------------------------------------------------

::

    [ 66%] Building NVCC (Device) object thrustrap/tests/CMakeFiles/TBufTest.dir/TBufTest_generated_TBufTest.cu.o
    /Users/blyth/opticks/thrustrap/tests/TBufTest.cu:145:226: warning: address of 'consoleAppender' will always evaluate to 'true' [-Wpointer-bool-conversion]
    { PLOG _plog(argc, argv); static plog::RollingFileAppender< plog::TxtFormatter>  fileAppender(_plog.logpath, _plog.logmax); static plog::ConsoleAppender< plog::TxtFormatter>  consoleAppender; { plog::IAppender *appender1 = (&consoleAppender) ? static_cast< plog::IAppender *>(&consoleAppender) : (__null); plog::IAppender *appender2 = (&fileAppender) ? static_cast< plog::IAppender *>(&fileAppender) : (__null); plog::Severity severity = static_cast< plog::Severity>(_plog.level); plog::init(severity, appender1); if (appender2) { plog::get()->addAppender(appender2); }  } ; } ; 
                                                                                                                                                                                                                                     ^~~~~~~~~~~~~~~  ~
    /Users/blyth/opticks/thrustrap/tests/TBufTest.cu:145:338: warning: address of 'fileAppender' will always evaluate to 'true' [-Wpointer-bool-conversion]
    { PLOG _plog(argc, argv); static plog::RollingFileAppender< plog::TxtFormatter>  fileAppender(_plog.logpath, _plog.logmax); static plog::ConsoleAppender< plog::TxtFormatter>  consoleAppender; { plog::IAppender *appender1 = (&consoleAppender) ? static_cast< plog::IAppender *>(&consoleAppender) : (__null); plog::IAppender *appender2 = (&fileAppender) ? static_cast< plog::IAppender *>(&fileAppender) : (__null); plog::Severity severity = static_cast< plog::Severity>(_plog.level); plog::init(severity, appender1); if (appender2) { plog::get()->addAppender(appender2); }  } ; } ; 
                                                                                                                                                                                                                                                                                                                                                     ^~~~~~~~~~~~  ~
    2 warnings generated.



First Use of CUDA App::seedPhotonsFromGensteps is slow ? A repeating without recompilation is faster
------------------------------------------------------------------------------------------------------

* presumably some compilation caching is being done 

* perhaps some nvcc compiler options are not correct,
  forcing compilation to the actual architecture at startup ?  YEP THIS LOOKS CORRECT

* http://stackoverflow.com/questions/23264229/nvidia-cuda-thrust-device-vector-allocation-is-too-slow

Initially tried changing CMakeLists.txt::

    +set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
    VERBOSE=1 thrap--

But that gives::

    [2015-Sep-17 10:59:52.039536]: App::seedPhotonsFromGensteps
    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: function_attributes(): after cudaFuncGetAttributes: invalid device function
    /Users/blyth/env/graphics/ggeoview/ggeoview.bash: line 1144: 29977 Abort trap: 6           $bin $*

Realising that "_20" is for Fermi not Kepler "_30" correcting options seems to fix invalid device function and slow first run problems::

     50 CUDA_ADD_LIBRARY( ${name}  
     51        TBuf_.cu
     52        TBufPair_.cu
     53        TSparse_.cu
     54        OPTIONS -gencode=arch=compute_30,code=sm_30
     55 )


link issue
~~~~~~~~~~~~

::

    Scanning dependencies of target TBuf4x4Test
    [ 69%] Linking CXX executable TBuf4x4Test
    Undefined symbols for architecture x86_64:
      "void TBuf::dump<float4>(char const*, unsigned int, unsigned int, unsigned int) const", referenced from:
          test_dump44() in TBuf4x4Test_generated_TBuf4x4Test.cu.o
      "void TBuf::dump<float4x4>(char const*, unsigned int, unsigned int, unsigned int) const", referenced from:
          test_dump44() in TBuf4x4Test_generated_TBuf4x4Test.cu.o
          test_dump4x4() in TBuf4x4Test_generated_TBuf4x4Test.cu.o
          test_count4x4() in TBuf4x4Test_generated_TBuf4x4Test.cu.o
          test_count4x4_ptr() in TBuf4x4Test_generated_TBuf4x4Test.cu.o
          test_copy4x4() in TBuf4x4Test_generated_TBuf4x4Test.cu.o
          test_copy4x4_ptr() in TBuf4x4Test_generated_TBuf4x4Test.cu.o
    ld: symbol(s) not found for architecture x86_64
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    make[2]: *** [thrustrap/tests/TBuf4x4Test] Error 1


Other packages using CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:env blyth$ find . -name CMakeLists.txt -exec grep -l CUDA {} \;
    ./cuda/cudawrap/CMakeLists.txt
    ./graphics/ggeoview/CMakeLists.txt
    ./graphics/optixrap/CMakeLists.txt
    ./graphics/raytrace/CMakeLists.txt
    ./graphics/thrust_opengl_interop/CMakeLists.txt
    ./thrust/hello/CMakeLists.txt
    ./thrustrap/CMakeLists.txt
    ./optix/gloptixthrust/CMakeLists.txt
    ./optix/OptiXTest/CMakeLists.txt
    ./optix/optixthrust/CMakeLists.txt
    ./optix/optixthrustnpy/CMakeLists.txt
    ./optix/optixthrustuse/CMakeLists.txt
    simon:env blyth$ 

Adjusted OPTIONS in 

* thrap-
* cudawrap-
* optixrap-

The others are testing only.   

TODO: centralize such settings



CUDA 5.5, Thrust and C++11 on Mavericks
------------------------------------------

Observations from the below match my experience

* https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks

CUDA 5.5 has problems with c++11 ie libc++ on Mavericks, 
can only get to compile and run without segv only by 

* targetting the older libstdc++::

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -stdlib=libstdc++")

* not linking to other libs built against c++11 ie libc++

delta:tests blyth$ otool -L /usr/local/env/thrustrap/bin/ThrustEngineTest
/usr/local/env/thrustrap/bin/ThrustEngineTest:
    @rpath/libcudart.5.5.dylib (compatibility version 0.0.0, current version 5.5.28)
    /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 60.0.0)
    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1197.1.1)


CUDA 7 to the rescue ? Yep upgrading to 7.0 fixes this
--------------------------------------------------------

* http://devblogs.nvidia.com/parallelforall/cuda-7-release-candidate-feature-overview/


Can a C interface provide a firewall to allow interop between compilers ?
----------------------------------------------------------------------------

Nope.  

The solution to get CUDA 5.5 to work with libc++ is to 
not use any C++ STL features like std::string.

*cudawrap-* did this, it uses C style string handling.

This is not an option with Thrust with is based on the STL.


CUDA OpenGL thrust interop
----------------------------

* https://gist.github.com/dangets/2926425

::

    #include <cuda_gl_interop.h>
    #include <thrust/device_vector.h>


    // initialization

    GLuint vbo;
    struct cudaGraphicsResource *vbo_cuda;

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo, cudaGraphicsMapFlagsWriteDiscard);


    // display time : handover OpenGL -> CUDA/thrust 

    cudaGraphicsMapResources(/*count*/1, &vbo_cuda,/*stream*/ 0);

    float4 *raw_ptr;
    size_t buf_size;
    cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, vbo_cuda);

    thrust::device_ptr<float4> dev_ptr = thrust::device_pointer_cast(raw_ptr);
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(g_mesh_width * g_mesh_height);
    thrust::transform(first, last, dev_ptr, sine_wave(g_mesh_width, g_mesh_height, g_anim));
 
    cudaGraphicsUnmapResources(1, &vbo_cuda, 0);  // CUDA/thrust back -> OpenGL
 


OptiX / OpenGL interop
------------------------


::

    OptiXEngine::init creates OptiX buffers using OpenGL buffer_id 

    m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, genstep_buffer_id);


    GMergedMeshOptiXGeometry uses OptiX buffer map/unmap when copying data into buffers


/Developer/OptiX/include/optixu/optixpp_namespace.h::

    1485     /// Get the pointer to buffer memory on a specific device. See @ref rtBufferGetDevicePointer
    1486     void getDevicePointer( unsigned int optix_device_number, CUdeviceptr *device_pointer );
    1487     CUdeviceptr getDevicePointer( unsigned int optix_device_number );



* https://devtalk.nvidia.com/default/topic/551556/?comment=3858139


thrust::stable_partition or thrust::copy_if
----------------------------------------------

* what size to allocate for the target ? do a reduce query to find out first, 
  or use stable_partition to shuffle 

* http://stackoverflow.com/questions/22371897/thrust-selectively-move-elements-to-another-vector

::

    thrust::device_vector<float>::iterator iter = thrust::stable_partition(A.begin(), A.end(), pred)
    thrust::device_vector<float> B(iter, A.end())
    A.erase(iter, A.end());




Histogramming Check
----------------------

::

    In [1]: h = phc_(1)
    INFO:env.g4dae.types:loading /usr/local/env/phcerenkov/1.npy 
    -rw-r--r--  1 blyth  staff  4902808 Jun 27 18:30 /usr/local/env/phcerenkov/1.npy

    In [2]: h[:,0,0]
    Out[2]: array([ 3265, 15297,     5, ...,     3,     3,     3], dtype=uint64)

    In [3]: hh = h[:,0,0]

   In [18]: uhh = np.unique(hh)

    In [19]: map(hex_, uhh)  # huh ? where the fffff from 
    Out[19]: 
    ['0x3',
     '0x5',
     '0x31',
     '0x51',
     '0x61',
     '0xc1',
     '0xf1',
     '0x361',
     '0x3b1',
     '0x3c1',
     '0x551',
     '0x561',
    ...
     '0x6cccc551',
     '0x6cccc561',
     '0x6cccc5c1',
     '0x6cccc651',
     '0x6ccccc51',
     '0x6cccccc1',
     '0xffffffffb55cc551',
     '0xffffffffb56ccc51',
     '0xffffffffb5b5c551',
     '0xffffffffb5bb5c51',
     '0xffffffffb5cc5c51',
     '0xffffffffb5cccc51',



      TODO:Compare with thrust...

      /usr/local/env/thrustrap/bin/PhotonIndexTest
    


Looks like no mapping/unmapping needed so long as dont change the size of the buffer


OpenGL buffer objects like PBOs and VBOs can be encapsulated for use in OptiX
with rtBufferCreateFromGLBO. The resulting buffer is a reference only to the
OpenGL data; the size of the OptiX buffer as well as the format have to be set
via rtBufferSetSize and rtBufferSetFormat. When the OptiX buffer is destroyed,
the state of the OpenGL buffer object is unaltered. Once an OptiX buffer is
created, the original GL buffer object is immutable, meaning the properties of
the GL object like its size cannot be changed while registered with OptiX.
However, it is still possible to read and write to the GL buffer object using
the appropriate GL functions. If it is necessary to change properties of an
object, first call rtBufferGLUnregister before making changes. After the
changes are made the object has to be registered again with rtBufferGLRegister.
This is necessary to allow OptiX to access the objects data again. Registration
and unregistration calls are expensive and should be avoided if possible.






EOU
}

thrap-env(){      olocal- ; opticks- ; }


thrap-sdir(){ echo $(opticks-home)/thrustrap ; }
thrap-tdir(){ echo $(opticks-home)/thrustrap/tests ; }
thrap-idir(){ echo $(opticks-idir) ; }
thrap-bdir(){ echo $(opticks-bdir)/thrustrap ; }

thrap-c(){    cd $(thrap-sdir); }
thrap-cd(){   cd $(thrap-sdir); }
thrap-scd(){  cd $(thrap-sdir); }
thrap-tcd(){  cd $(thrap-tdir); }
thrap-icd(){  cd $(thrap-idir); }
thrap-bcd(){  cd $(thrap-bdir); }

thrap-name(){ echo ThrustRap ; }
thrap-tag(){  echo THRAP ; }

thrap-apihh(){  echo $(thrap-sdir)/$(thrap-tag)_API_EXPORT.hh ; }
thrap---(){     touch $(thrap-apihh) ; thrap--  ; }


thrap-wipe(){ local bdir=$(thrap-bdir) ;  rm -rf $bdir ; }

thrap--(){                   opticks-- $(thrap-bdir) ; } 
thrap-t(){                   opticks-t $(thrap-bdir) $* ; } 
thrap-genproj() { thrap-scd ; opticks-genproj $(thrap-name) $(thrap-tag) ; } 
thrap-gentest() { thrap-tcd ; opticks-gentest ${1:-TExample} $(thrap-tag) ; } 
thrap-txt(){ vi $(thrap-sdir)/CMakeLists.txt $(thrap-tdir)/CMakeLists.txt ; } 



thrap-env(){  
   olocal- 
   cuda-
   cuda-export
   #optix-
   #optix-export
   thrust-
   thrust-export 
}

thrap-cmake-deprecated(){
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   local bdir=$(thrap-bdir)
   mkdir -p $bdir
  
   thrap-bcd 

   local flags=$(cuda-nvcc-flags)
   echo $msg using CUDA_NVCC_FLAGS $flags

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(thrap-idir) \
       -DCUDA_NVCC_FLAGS="$flags" \
       $(thrap-sdir)

   cd $iwd
}

thrap-bin(){ echo $(thrap-idir)/bin/$(thrap-name)Test ; }
thrap-export()
{ 
   echo -n 
}
thrap-run(){
   local bin=$(thrap-bin)
   thrap-export
   $bin $*
}




thrap-print()
{
    thrust_curand_printf $1
    curand_aligned_host $1
}


thrap-expandTest-touch(){ touch $(opticks-home)/thrustrap/tests/expandTest.cu  ; }

thrap-expandTest-p()
{
    cd $(opticks-prefix)/build/thrustrap/tests
    thrap-expandTest-touch
    make expandTest
    ./expandTest
}

thrap-expandTest-i()
{
    cd $(opticks-prefix-tmp)/build/thrustrap/tests
    thrap-expandTest-touch
    make expandTest
    ./expandTest
}

thrap-expandTest-notes(){ cat << EON

epsilon:tests blyth$ thrap-expandTest-p > /tmp/thrap-expandTest-p
Generated /usr/local/opticks-cmake-overhaul/build/thrustrap/tests/CMakeFiles/expandTest.dir//./expandTest_generated_expandTest.cu.o successfully.
epsilon:tests blyth$ thrap-expandTest-i > /tmp/thrap-expandTest-i
Generated /usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/tests/CMakeFiles/expandTest.dir//./expandTest_generated_expandTest.cu.o successfully.
libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: device free failed: CUDA driver version is insufficient for CUDA runtime version
Abort trap: 6

perl -pi -e 's,/usr/local/opticks-cmake-overhaul-tmp,,g' /tmp/thrap-expandTest-i

EON
}





thrap-expandTest-pp(){

# copy pasting from: cat /tmp/thrap-expandTest-p | CMakeOutput.py


/Developer/NVIDIA/CUDA-9.1/bin/nvcc \
/Users/blyth/opticks-cmake-overhaul/thrustrap/tests/expandTest.cu \
-c \
-o \
/usr/local/opticks-cmake-overhaul/build/thrustrap/tests/CMakeFiles/expandTest.dir//./expandTest_generated_expandTest.cu.o \
-ccbin \
/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang \
-m64 \
-DWITH_YoctoGL \
-DWITH_ImplicitMesher \
-DWITH_DualContouringSample \
-Xcompiler \
-fPIC \
-gencode=arch=compute_30,code=sm_30 \
-std=c++11 \
-O2 \
--use_fast_math \
-DNVCC \
-I/Developer/NVIDIA/CUDA-9.1/include \
-I/Users/blyth/opticks-cmake-overhaul/thrustrap \
-I/usr/local/opticks-cmake-overhaul/include/OpticksCore \
-I/usr/local/opticks-cmake-overhaul/include/NPY \
-I/usr/local/opticks-cmake-overhaul/externals/glm/glm \
-I/usr/local/opticks-cmake-overhaul/include/SysRap \
-I/usr/local/opticks-cmake-overhaul/externals/plog/include \
-I/usr/local/opticks-cmake-overhaul/include/BoostRap \
-I/opt/local/include \
-I/usr/local/opticks-cmake-overhaul/externals/include \
-I/usr/local/opticks-cmake-overhaul/externals/include/YoctoGL \
-I/usr/local/opticks-cmake-overhaul/externals/include/DualContouringSample \
-I/usr/local/opticks-cmake-overhaul/include/OKConf \
-I/usr/local/opticks-cmake-overhaul/include/CUDARap 



/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ \
 \
 \
-fvisibility=hidden \
-Wall \
-Wno-unused-function \
-Wno-unused-private-field \
-Wno-shadow \
-g \
-Wl,-search_paths_first \
-Wl,-headerpad_max_install_names \
 \
/usr/local/opticks-cmake-overhaul/build/thrustrap/tests/CMakeFiles/expandTest.dir/expandTest_generated_expandTest.cu.o \
 \
-o \
expandTest \
-Wl,-rpath,/usr/local/opticks-cmake-overhaul/lib \
-Wl,-rpath,/usr/local/opticks-cmake-overhaul/externals/lib \
-Wl,-rpath,/Developer/NVIDIA/CUDA-9.1/lib \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
-Wl,-rpath,/usr/local/cuda/lib \
/usr/local/opticks-cmake-overhaul/build/thrustrap/libThrustRap.dylib \
/usr/local/opticks-cmake-overhaul/lib/libOpticksCore.dylib \
/usr/local/opticks-cmake-overhaul/lib/libNPY.dylib \
/usr/local/opticks-cmake-overhaul/lib/libBoostRap.dylib \
/opt/local/lib/libboost_program_options-mt.dylib \
/opt/local/lib/libboost_filesystem-mt.dylib \
/opt/local/lib/libboost_system-mt.dylib \
/opt/local/lib/libboost_regex-mt.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libOpenMeshCore.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libOpenMeshTools.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libYoctoGL.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libDualContouringSample.dylib \
/usr/local/opticks-cmake-overhaul/lib/libOKConf.dylib \
/usr/local/opticks-cmake-overhaul/lib/libCUDARap.dylib \
/usr/local/opticks-cmake-overhaul/lib/libSysRap.dylib \
-lcrypto \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
/Developer/NVIDIA/CUDA-9.1/lib/libcurand.dylib \
-lssl 


./expandTest    ## this succeeds to run 


}



thrap-expandTest-ii(){

# copy pasting from: cat /tmp/thrap-expandTest-i | CMakeOutput.py
# fails like in integrated build

/Developer/NVIDIA/CUDA-9.1/bin/nvcc \
/Users/blyth/opticks-cmake-overhaul/thrustrap/tests/expandTest.cu \
-c \
-o \
/usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/tests/CMakeFiles/expandTest.dir//./expandTest_generated_expandTest.cu.o \
-ccbin \
/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang \
-m64 \
-DWITH_YoctoGL \
-DWITH_ImplicitMesher \
-DWITH_DualContouringSample \
-Xcompiler \
-fPIC \
-gencode=arch=compute_30,code=sm_30 \
-std=c++11 \
-O2 \
--use_fast_math \
-DNVCC \
-I/Users/blyth/opticks-cmake-overhaul/thrustrap \
-I/Users/blyth/opticks-cmake-overhaul/optickscore \
-I/Users/blyth/opticks-cmake-overhaul/npy \
-I/usr/local/opticks-cmake-overhaul-tmp/externals/glm/glm \
-I/Users/blyth/opticks-cmake-overhaul/sysrap \
-I/usr/local/opticks-cmake-overhaul-tmp/externals/plog/include \
-I/Users/blyth/opticks-cmake-overhaul/boostrap \
-I/usr/local/opticks-cmake-overhaul-tmp/build/boostrap/inc \
-I/opt/local/include \
-I/usr/local/opticks-cmake-overhaul/externals/include \
-I/usr/local/opticks-cmake-overhaul/externals/include/YoctoGL \
-I/usr/local/opticks-cmake-overhaul/externals/include/DualContouringSample \
-I/usr/local/opticks-cmake-overhaul-tmp/build/okconf/inc \
-I/Users/blyth/opticks-cmake-overhaul/okconf \
-I/Users/blyth/opticks-cmake-overhaul/cudarap \
-I/Developer/NVIDIA/CUDA-9.1/include 



cd /usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/tests

/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ \
 \
 \
-fvisibility=hidden \
-Wall \
-Wno-unused-function \
-Wno-unused-private-field \
-Wno-shadow \
-g \
-Wl,-search_paths_first \
-Wl,-headerpad_max_install_names \
 \
CMakeFiles/expandTest.dir/expandTest_generated_expandTest.cu.o \
 \
-o \
expandTest \
-Wl,-rpath,/usr/local/opticks-cmake-overhaul-tmp/lib \
-Wl,-rpath,/usr/local/opticks-cmake-overhaul/externals/lib \
-Wl,-rpath,/Developer/NVIDIA/CUDA-9.1/lib \
../libThrustRap.dylib \
../../optickscore/libOpticksCore.dylib \
../../npy/libNPY.dylib \
../../boostrap/libBoostRap.dylib \
/opt/local/lib/libboost_program_options-mt.dylib \
/opt/local/lib/libboost_filesystem-mt.dylib \
/opt/local/lib/libboost_system-mt.dylib \
/opt/local/lib/libboost_regex-mt.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libOpenMeshCore.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libOpenMeshTools.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libYoctoGL.dylib \
/usr/local/opticks-cmake-overhaul/externals/lib/libDualContouringSample.dylib \
../../okconf/libOKConf.dylib \
../../cudarap/libCUDARap.dylib \
../../sysrap/libSysRap.dylib \
-lcrypto \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
/Developer/NVIDIA/CUDA-9.1/lib/libcurand.dylib \
-lssl 


./expandTest


}



thrap-expandTest-pp-short(){

/Developer/NVIDIA/CUDA-9.1/bin/nvcc \
/Users/blyth/opticks-cmake-overhaul/thrustrap/tests/expandTest.cu \
-c \
-o \
/usr/local/opticks-cmake-overhaul/build/thrustrap/tests/CMakeFiles/expandTest.dir//./expandTest_generated_expandTest.cu.o \
-ccbin \
/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang \
-m64 \
-Xcompiler \
-fPIC \
-gencode=arch=compute_30,code=sm_30 \
-std=c++11 \
-O2 \
--use_fast_math \
-DNVCC \
-I/Developer/NVIDIA/CUDA-9.1/include \
-I/Users/blyth/opticks-cmake-overhaul/thrustrap 

/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ \
 \
 \
-fvisibility=hidden \
-Wall \
-Wno-unused-function \
-Wno-unused-private-field \
-Wno-shadow \
-g \
-Wl,-search_paths_first \
-Wl,-headerpad_max_install_names \
 \
/usr/local/opticks-cmake-overhaul/build/thrustrap/tests/CMakeFiles/expandTest.dir/expandTest_generated_expandTest.cu.o \
 \
-o \
expandTest \
-Wl,-rpath,/usr/local/cuda/lib \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a 

./expandTest   


cat << EON

The ii problem can be made to occur for these pp commands  
by changing the library and rpath setup in the below 
original 10 pp lines (that works)::

-Wl,-rpath,/usr/local/opticks-cmake-overhaul/lib \
-Wl,-rpath,/usr/local/opticks-cmake-overhaul/externals/lib \
-Wl,-rpath,/Developer/NVIDIA/CUDA-9.1/lib \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
-Wl,-rpath,/usr/local/cuda/lib \
 \
-lcrypto \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
/Developer/NVIDIA/CUDA-9.1/lib/libcurand.dylib \
-lssl 

failing::

/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
-Wl,-rpath,/Developer/NVIDIA/CUDA-9.1/lib 

works::

-Wl,-rpath,/usr/local/cuda/lib \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a 

works::

/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
-Wl,-rpath,/usr/local/cuda/lib 

also works::

/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
-Wl,-rpath,/Developer/NVIDIA/CUDA-9.1/lib \
-Wl,-rpath,/usr/local/cuda/lib 

shortened that still works::

-Wl,-rpath,/Developer/NVIDIA/CUDA-9.1/lib \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
-Wl,-rpath,/usr/local/cuda/lib 


observations
     crypto and ssl and curand not needed


seems that require the below otherwise nogo

   -Wl,-rpath,/usr/local/cuda/lib  


epsilon:tests blyth$ otool-rpath expandTest 
          cmd LC_RPATH
      cmdsize 32
         path /usr/local/cuda/lib (offset 12)
epsilon:tests blyth$ 


epsilon:tests blyth$ realpath /Developer/NVIDIA/CUDA-9.1/lib
/Developer/NVIDIA/CUDA-9.1/lib
epsilon:tests blyth$ realpath /usr/local/cuda/lib
/usr/local/cuda/lib
epsilon:tests blyth$ l /usr/local/cuda/lib/
total 32
lrwxr-xr-x  1 root  wheel  -    50 Dec 20 19:54 libcublas.9.1.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libcublas.9.1.dylib
lrwxr-xr-x  1 root  wheel  -    46 Dec 20 19:54 libcublas.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libcublas.dylib
lrwxr-xr-x  1 root  wheel  -    49 Dec 20 19:54 libcublas_device.a -> /Developer/NVIDIA/CUDA-9.1/lib/libcublas_device.a
lrwxr-xr-x  1 root  wheel  -    49 Dec 20 19:54 libcublas_static.a -> /Developer/NVIDIA/CUDA-9.1/lib/libcublas_static.a
lrwxr-xr-x  1 root  wheel  -    45 Dec 20 19:54 libcudadevrt.a -> /Developer/NVIDIA/CUDA-9.1/lib/libcudadevrt.a
lrwxr-xr-x  1 root  wheel  -    50 Dec 20 19:54 libcudart.9.1.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libcudart.9.1.dylib
lrwxr-xr-x  1 root  wheel  -    46 Dec 20 19:54 libcudart.dylib -> /Developer/NVIDIA/CUDA-9.1/lib/libcudart.dylib
lrwxr-xr-x  1 root  wheel  -    49 Dec 20 19:54 libcudart_static.a -> /Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a


given than /usr/local/cuda/lib/ is comprised of a bunch of links to libs in 
/Developer/NVIDIA/CUDA-9.1/lib  the fail is kinda surprising ... 


EON


}

thrap-expandTest-ii-short(){    ## this one doesnt work

/Developer/NVIDIA/CUDA-9.1/bin/nvcc \
/Users/blyth/opticks-cmake-overhaul/thrustrap/tests/expandTest.cu \
-c \
-o \
/usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/tests/CMakeFiles/expandTest.dir//./expandTest_generated_expandTest.cu.o \
-ccbin \
/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang \
-m64 \
-Xcompiler \
-fPIC \
-gencode=arch=compute_30,code=sm_30 \
-std=c++11 \
-O2 \
--use_fast_math \
-DNVCC \
-I/Users/blyth/opticks-cmake-overhaul/thrustrap \
-I/Developer/NVIDIA/CUDA-9.1/include 

## CUDA last from integrated is the one that doesnt work, but putting it first does not fix

## only difference is header order, CUDA last 


cd /usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/tests

/Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ \
 \
 \
-fvisibility=hidden \
-Wall \
-Wno-unused-function \
-Wno-unused-private-field \
-Wno-shadow \
-g \
-Wl,-search_paths_first \
-Wl,-headerpad_max_install_names \
 \
CMakeFiles/expandTest.dir/expandTest_generated_expandTest.cu.o \
 \
-o \
expandTest \
-Wl,-rpath,/usr/local/opticks-cmake-overhaul-tmp/lib \
-Wl,-rpath,/usr/local/opticks-cmake-overhaul/externals/lib \
-Wl,-rpath,/usr/local/cuda/lib \
-lcrypto \
/Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
/Developer/NVIDIA/CUDA-9.1/lib/libcurand.dylib \
-lssl 

./expandTest


cat << EON


 try swapping 
    -Wl,-rpath,/Developer/NVIDIA/CUDA-9.1/lib \
 for 
    -Wl,-rpath,/usr/local/cuda/lib \


  YEP : doing this swap succeed to get ii to run ....

EON

}












thrap-glm-test-notes(){ cat << EON

The "dereferencing type-punned pointer will break strict-aliasing rules" warnings appear only with optimisation "-O2"

home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl: In function ‘glm::uint glm::packUnorm2x16(const vec2&)’:
/home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl:42:46: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
   return reinterpret_cast<uint const &>(Topack);
  
See also glm-nvcc and 

* notes/issues/glm-0.9.6.3-nvcc-warnings-dereferencing-type-punned-pointer-will-break-strict-aliasing-rules.rst

EON
}


thrap-glm-warning(){ cat << EOW

/usr/local/cuda-10.1/bin/nvcc /home/blyth/opticks/thrustrap/TBuf_.cu -c -o /home/blyth/local/opticks/build/thrustrap/CMakeFiles/ThrustRap.dir//./ThrustRap_generated_TBuf_.cu.o -ccbin /usr/bin/cc \
       -m64 \
         -DThrustRap_EXPORTS -DOPTICKS_THRAP -DOPTICKS_OKCORE -DOPTICKS_NPY -DOPTICKS_YoctoGL -DOPTICKS_ImplicitMesher -DOPTICKS_DualContouringSample -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -DOPTICKS_BRAP -DOPTICKS_CUDARAP \
         -Xcompiler ,\"-fvisibility=hidden\",\"-fvisibility-inlines-hidden\",\"-Wall\",\"-Wno-unused-function\",\"-Wno-comment\",\"-Wno-deprecated\",\"-Wno-shadow\",\"-fPIC\",\"-g\" \
          -Xcompiler -fPIC -gencode=arch=compute_70,code=sm_70 -O2 \
         --use_fast_math -DNVCC \
             -I/usr/local/cuda-10.1/include \
             -I/home/blyth/opticks/thrustrap \
             -I/home/blyth/local/opticks/include/OpticksCore \
             -I/home/blyth/local/opticks/include/NPY \
             -I/home/blyth/local/opticks/externals/glm/glm \
             -I/home/blyth/local/opticks/include/SysRap \
             -I/home/blyth/local/opticks/externals/plog/include \
             -I/home/blyth/local/opticks/include/OKConf \
             -I/home/blyth/local/opticks/include/BoostRap \
             -I/usr/include \
             -I/home/blyth/local/opticks/externals/include \
             -I/home/blyth/local/opticks/externals/include/YoctoGL \
             -I/home/blyth/local/opticks/externals/include/DualContouringSample \
             -I/home/blyth/local/opticks/include/CUDARap \
             -I/usr/local/cuda-10.1/samples/common/inc

EOW
}


thrap-glm-test-(){ cat << EOT

#include <iostream>
#include <glm/glm.hpp>

int main(int argc, char** argv)
{
    //glm::vec2 v(1.f, 2.f) ; 
    //glm::uint u = glm::packUnorm2x16(v) ;  
    //std::cout << " u " << u << std::endl ; 
    return 0 ; 
}

EOT
} 

thrap-glm-test(){ 
   local tmp=/tmp/$USER/opticks/$FUNCNAME
   local iwd=$PWD
   rm -rf $tmp && mkdir -p $tmp && cd $tmp

   $FUNCNAME- > $FUNCNAME.cu

    nvcc $FUNCNAME.cu \
       -m64 \
         -Xcompiler ,\"-fvisibility=hidden\",\"-fvisibility-inlines-hidden\",\"-Wall\",\"-Wno-unused-function\",\"-Wno-comment\",\"-Wno-deprecated\",\"-Wno-shadow\",\"-fPIC\",\"-g\" \
          -Xcompiler -fPIC \
            -O2 \
          -gencode=arch=compute_70,code=sm_70 \
         --use_fast_math -DNVCC \
        -I$LOCAL_BASE/opticks/externals/glm/glm
}








