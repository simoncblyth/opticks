helper_cuda : Ben reports that helper_cuda.h is not present in the docker CUDA distrib 
=========================================================================================




::

    epsilon:opticks blyth$ opticks-find helper_cuda.h
    ./cudarap/CResource_.cu:#include "helper_cuda.h"
    ./examples/UseOpticksCUDA/UseOpticksCUDA.cu:#include "helper_cuda.h"
    ./cudarap/CMakeLists.txt:Formerly copied in helper_cuda.h from the samples distrib, now trying to avoid 
    ./cudarap/CMakeLists.txt:  /Developer/NVIDIA/CUDA-9.1/samples/common/inc/helper_cuda.h
    ./cudarap/CMakeLists.txt:  /Volumes/Delta/Developer/NVIDIA/CUDA-7.0/samples/common/inc/helper_cuda.h 
    ./cudarap/CMakeLists.txt:  /Volumes/Delta/Developer/NVIDIA/CUDA-5.5/samples/common/inc/helper_cuda.h 



    epsilon:opticks blyth$ cuda-cd
    epsilon:CUDA-9.1 blyth$ find . -name helper_cuda.h 
    ./samples/common/inc/helper_cuda.h
    epsilon:CUDA-9.1 blyth$ pwd
    /Developer/NVIDIA/CUDA-9.1

    epsilon:NVIDIA blyth$ cd /Volumes/Delta/Developer/NVIDIA/
    epsilon:NVIDIA blyth$ l
    total 0
    drwxrwxrwx  15 root  wheel  510 Jun 29  2015 CUDA-7.0
    drwxrwxrwx  16 root  wheel  544 Jan 15  2014 CUDA-5.5

    epsilon:NVIDIA blyth$ find . -name helper_cuda.h 
    ./CUDA-5.5/samples/common/inc/helper_cuda.h
    ./CUDA-7.0/samples/common/inc/helper_cuda.h
    epsilon:NVIDIA blyth$ 




