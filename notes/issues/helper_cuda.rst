helper_cuda : Ben reports that helper_cuda.h is not present in the docker CUDA distrib 
=========================================================================================

helper_cuda.h and helper_string.h are used only in a few places for converting error enum
to strings and for an error checking macro. They are part of the CUDA SDK in the samples/common/inc directory.::

    epsilon:opticks blyth$ opticks-find helper_cuda.h
    ./cudarap/CResource_.cu:#include "helper_cuda.h"   // for checkCudaErrors
    ./examples/UseOpticksCUDA/UseOpticksCUDA.cu:#include "helper_cuda.h"    // for _cudaGetErrorEnum
    ./cudarap/CMakeLists.txt:Formerly copied in helper_cuda.h from the samples distrib, now trying to avoid 
    ./cudarap/CMakeLists.txt:  /Developer/NVIDIA/CUDA-9.1/samples/common/inc/helper_cuda.h
    ./cudarap/CMakeLists.txt:  /Volumes/Delta/Developer/NVIDIA/CUDA-7.0/samples/common/inc/helper_cuda.h 
    ./cudarap/CMakeLists.txt:  /Volumes/Delta/Developer/NVIDIA/CUDA-5.5/samples/common/inc/helper_cuda.h 


To workaround these not being present in the CUDA docker distrib, I have included 
some fallback versions in cmake/Modules/include/helper_cuda_fallback/
And changed cmake/Modules/FindOpticksCUDA.cmake to look for them there if 
not found in standard place::

     24    # see notes/issues/helper_cuda.rst
     25    find_path(
     26        HELPER_CUDA_INCLUDE_DIR
     27        NAMES "helper_cuda.h"
     28        PATHS
     29            "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc"
     30            "${CMAKE_CURRENT_LIST_DIR}/include/helper_cuda_fallback/${CUDA_VERSION}"
     31     )
     32     if(HELPER_CUDA_INCLUDE_DIR)
     33         set(OpticksHELPER_CUDA_FOUND "YES")
     34     else()
     35         set(OpticksHELPER_CUDA_FOUND "NO")
     36     endif()
     37 

For definiteness I have put them in a directory with the CUDA version. However  
they change only slowly with CUDA version. Newer CUDA versions  
will probably work with older helpers. 

::

    epsilon:~ blyth$ o
    M cmake/Modules/FindOpticksCUDA.cmake
    M cudarap/CResource_.cu
    M examples/UseOpticksCUDA/UseOpticksCUDA.cu
    M notes/issues/OpticksCUDAFlags.rst
    M notes/issues/helper_cuda.rst
    A cmake/Modules/include/helper_cuda_fallback/9.1/helper_cuda.h
    A cmake/Modules/include/helper_cuda_fallback/9.1/helper_string.h
    epsilon:opticks blyth$ 


To test::

   [blyth@localhost opticks]$ cd examples/UseOpticksCUDA
   [blyth@localhost UseOpticksCUDA]$ ./go.sh 
   ...

    -- Found CUDA: /usr/local/cuda-9.2 (found version "9.2") 
    -- Use examples/UseOpticksCUDA/CMakeLists.txt for testing FindOpticksCUDA.cmake
    --   CUDA_TOOLKIT_ROOT_DIR   : /usr/local/cuda-9.2 
    --   CUDA_SDK_ROOT_DIR       : CUDA_SDK_ROOT_DIR-NOTFOUND 
    --   CUDA_VERSION            : 9.2 
    --   HELPER_CUDA_INCLUDE_DIR : /usr/local/cuda-9.2/samples/common/inc 
    --   PROJECT_SOURCE_DIR      : /home/blyth/opticks/examples/UseOpticksCUDA 
    --   CMAKE_CURRENT_LIST_DIR  : /home/blyth/opticks/cmake/Modules 
    -- FindOpticksCUDA.cmake:OpticksCUDA_VERBOSE      : ON 
    -- FindOpticksCUDA.cmake:OpticksCUDA_FOUND        : YES 
    -- FindOpticksCUDA.cmake:OpticksHELPER_CUDA_FOUND : YES 
    -- FindOpticksCUDA.cmake:OpticksCUDA_API_VERSION  : 9020 
    -- FindOpticksCUDA.cmake:CUDA_LIBRARIES           : /usr/local/cuda-9.2/lib64/libcudart_static.a;-lpthread;dl;/usr/lib64/librt.so 
    -- FindOpticksCUDA.cmake:CUDA_INCLUDE_DIRS        : /usr/local/cuda-9.2/include 
    -- FindOpticksCUDA.cmake:CUDA_curand_LIBRARY      : /usr/local/cuda-9.2/lib64/libcurand.so
     key='CUDA_cudart_static_LIBRARY' val='/usr/local/cuda-9.2/lib64/libcudart_static.a' 
     key='CUDA_curand_LIBRARY' val='/usr/local/cuda-9.2/lib64/libcurand.so' 

    -- Configuring done





Hi Raja, 


> On Jan 20, 2021, at 2:54 PM, Raja Nandakumar <rajanandakumar@gmail.com> wrote:
> Hi,
>
> When I try to build opticks-full, I get the following error when it goes through cudarap :
>
> CMake Error at tests/CMakeLists.txt:17 (add_executable):
>  Target "cuRANDWrapperTest" links to target "Opticks::CUDASamples" but the
>  target was not found.  Perhaps a find_package() call is missing for an
>  IMPORTED target, or an ALIAS target is missing?
>
> I try to get around this by setting
> 
> export CUDA_SAMPLES=/usr/local/cuda/cuda-samples
> 
> At this point I hit the error
>
> [  4%] Linking CXX shared library libCUDARap.so
> /usr/bin/ld: cannot find -lOpticks::CUDASamples
> collect2: error: ld returned 1 exit status
> 
> What would I be doing wrong here?
>
> Thanks and Cheers,
> Raja.
> _._,_._,_


Note that for problems like this it is 
helpful to use the standalone example::

 
     cd ~/opticks/examples/UseOpticksCUDA
     ./go.sh 


In cudarap/CMakeLists.txt notice line 15

 14 set(OpticksCUDA_VERBOSE ON)
 15 find_package(OpticksCUDA REQUIRED MODULE)

That line leads to the loading of cmake/Modules/FindOpticksCUDA.cmake
The line 121 below is where the Opticks::CUDASamples target should be created, 
but clearly it needs OpticksHELPER_CUDA_FOUND


cmake/Modules/FindOpticksCUDA.cmake:

102 if(OpticksCUDA_FOUND AND NOT TARGET ${tgt})
103     add_library(Opticks::cudart_static UNKNOWN IMPORTED)
104     set_target_properties(Opticks::cudart_static PROPERTIES IMPORTED_LOCATION "${CUDA_cudart_static_LIBRARY}")
105     set_target_properties(Opticks::cudart_static PROPERTIES INTERFACE_IMPORTED_LOCATION "${CUDA_cudart_static_LIBRARY}")
106     # duplicate with INTERFACE_ to workaround CMake 3.13 whitelisting restriction
107 
108     add_library(Opticks::curand UNKNOWN IMPORTED)
109     set_target_properties(Opticks::curand PROPERTIES IMPORTED_LOCATION "${CUDA_curand_LIBRARY}")
110     set_target_properties(Opticks::curand PROPERTIES INTERFACE_IMPORTED_LOCATION "${CUDA_curand_LIBRARY}")
111     # duplicate with INTERFACE_ to workaround CMake 3.13 whitelisting restriction
112 
113     add_library(${tgt} INTERFACE IMPORTED)
114     set_target_properties(${tgt}  PROPERTIES INTERFACE_FIND_PACKAGE_NAME "OpticksCUDA MODULE REQUIRED")
115     set_target_properties(${tgt}  PROPERTIES INTERFACE_PKG_CONFIG_NAME   "OpticksCUDA")
116 
117     target_link_libraries(${tgt} INTERFACE Opticks::cudart_static Opticks::curand )
118     target_include_directories(${tgt} INTERFACE "${CUDA_INCLUDE_DIRS}" )
119 
120     if(OpticksHELPER_CUDA_FOUND)
121         add_library(Opticks::CUDASamples INTERFACE IMPORTED)
122         target_include_directories(Opticks::CUDASamples INTERFACE "${HELPER_CUDA_INCLUDE_DIR}")
123         ## for CUDA error strings from helper_cuda.h and helper_string.h 
124     endif()
125 
126     set(OpticksCUDA_targets
127          cudart_static
128          curand
129          CUDA
130          CUDASamples
131     )
132 endif()


Looking for OpticksHELPER_CUDA_FOUND I see that line 59 is the critical one.
This means that CMake presumably fails find a file called “helper_cuda.h”  
There is a reference to some notes about helper_cuda on line 58 

    notes/issues/helper_cuda.rst 
    https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/helper_cuda.rst

That explains the reason for the helper_cuda_fallback on line 64.


 46 set(CUDA_API_VERSION_INTEGER 0)
 47 if(OpticksCUDA_FOUND)
 48    file(READ "${CUDA_INCLUDE_DIRS}/cuda.h" _contents)
 49    string(REGEX REPLACE "\n" ";" _contents "${_contents}")
 50    foreach(_line ${_contents})
 51        #if (_line MATCHES "^    #define __CUDA_API_VERSION ([0-9]+)") ## require 4 spaces to distinguish from another ancient API version 
 52        if (_line MATCHES "#define CUDA_VERSION ([0-9]+)") ## require 4 spaces to distinguish from another ancient API version 
 53             set(OpticksCUDA_API_VERSION ${CMAKE_MATCH_1} )
 54             #message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_API_VERSION:${OpticksCUDA_API_VERSION}") 
 55        endif()
 56    endforeach()
 57 
 58    # see notes/issues/helper_cuda.rst
 59    find_path(
 60        HELPER_CUDA_INCLUDE_DIR
 61        NAMES "helper_cuda.h"
 62        PATHS
 63            "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc"
 64            "${CMAKE_CURRENT_LIST_DIR}/include/helper_cuda_fallback/${CUDA_VERSION}"
 65     )
 66     if(HELPER_CUDA_INCLUDE_DIR)
 67         set(OpticksHELPER_CUDA_FOUND "YES")
 68     else()
 69         set(OpticksHELPER_CUDA_FOUND "NO")
 70     endif()
 71 
 72 endif()




Checking the CUDA versions that have the fallback I see::

epsilon:opticks blyth$ l ./cmake/Modules/include/helper_cuda_fallback/
total 0
drwxr-xr-x  4 blyth  staff  128 May 16  2020 9.2
drwxr-xr-x  4 blyth  staff  128 May 16  2020 9.1
epsilon:opticks blyth$ 

epsilon:opticks blyth$ l ./cmake/Modules/include/helper_cuda_fallback/9.2/
total 144
-rw-r--r--  1 blyth  staff  33798 May 16  2020 helper_string.h
-rw-r--r--  1 blyth  staff  36542 May 16  2020 helper_cuda.h
epsilon:opticks blyth$ 


So it would seem that your version or distribution of CUDA does not 
have the requisite helper_cuda.h or perhaps it does but it but in a moved location.
Also, presumably you are not using 9.1 or 9.2 for which there are fallbacks.

So the question is : what CUDA version are you using ? 
And how did you install it ?

To fix the issue to support your CUDA version I will need to try out 
this version and if the helper_cuda.h is no longer provided I will have to 
create a suitable fallback.


Notice that the version of CUDA you should be using is tied to the 
version of OptiX that you are using. 
Because Opticks compiles against both OptiX and CUDA it is recommended to 
use precisely the CUDA version that the OptiX version was built with
as stated in the OptiX release notes.

If you are using OptiX 6.5 then you should be using CUDA 10.1 as that is 
what was used to build OptiX 6.5 

If you find problems when using other version combinations please
report your findings in copy/paste detail.


Simon




