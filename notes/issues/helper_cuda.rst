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






