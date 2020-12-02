cudarap_cudaMalloc_fail
=========================

:: 

    Dear Simon,

    thanks a lot, now it is compiled well, but it gives an error which is related
    to writing info to video card (as I think), later I will try to fix it and in a
    case of no success I will ask you again.

    Error for the final installation of opticks:

    [opticks-make] 2020-12-01 19:42:59.335 INFO  [818480] [main@54]  work 10000000 max_blocks 128 seed 0 offset 0 threads_per_block 256 cachedir /home/chukanov/.opticks/rngcache/RNG
    [opticks-make] 2020-12-01 19:42:59.335 INFO  [818480] [allocate_rng_wrapper@68]  items  10000000 items/M 10 sizeof(curandState) 48 nbytes 480000000 nbytes/M 480
    [opticks-make] Cuda error in file '/cern/juno/ExternalLibs/Build/opticks-0.0.0-rc3/cudarap/cuRANDWrapper_kernel.cu' in line 78 : unknown error.
    [opticks-make] ====== juno-ext-libs-opticks-make- : ABORT opticks-full-make FAIL rc 1


     Also, please add to the file

    ExternalLibs/Opticks/0.0.0-rc3/externals/openmesh/OpenMesh-6.3/src/OpenMesh/Tools/Utils/conio.cc

    string:

    #include <sys/time.h>

    otherwise in Debian system with gcc >= 7 compilation gives an error.

     Best regards,

           Artem.


Looks like 10M curandState items (480M) is too much VRAM for this GPU perhaps ?:: 

     epsilon:junotop blyth$ vi ExternalLibs/Build/opticks-0.0.0-rc1/cudarap/cuRANDWrapper_kernel.cu  +78

     52 /**
     53 allocate_rng_wrapper
     54 ---------------------
     55 
     56 Allocates curandState device buffer sized to hold the number
     57 of items from LaunchSequence and returns pointer to it. 
     58 
     59 **/
     60 
     61 CUdeviceptr allocate_rng_wrapper( LaunchSequence* launchseq)
     62 {
     63     unsigned int items = launchseq->getItems();
     64     size_t nbytes = items*sizeof(curandState) ;
     65     int value = 0 ;
     66     int M = 1000000 ;
     67 
     68     LOG(info)
     69          << " items  " << items
     70          << " items/M " << items/M
     71          << " sizeof(curandState) " << sizeof(curandState)
     72          << " nbytes " << nbytes
     73          << " nbytes/M " << nbytes/M
     74          ;
     75 
     76     CUdeviceptr dev_rng_states ;
     77 
     78     CUDA_SAFE_CALL( cudaMalloc((void**)&dev_rng_states, nbytes ));
     79 
     80     CUDA_SAFE_CALL( cudaMemset((void*)dev_rng_states, value, nbytes ));
     81 
     82     return dev_rng_states ;
     83 }




Workaround : Add OPTICKS_CUDARAP_RNGMAX envvar with default of 1,3,10 that controls the curandStates inited
--------------------------------------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ git diff cudarap/cudarap.bash
    diff --git a/cudarap/cudarap.bash b/cudarap/cudarap.bash
    index 90aa196a..3d5dd134 100644
    --- a/cudarap/cudarap.bash
    +++ b/cudarap/cudarap.bash
    @@ -563,18 +563,9 @@ EON
     
     
     
    -cudarap-prepare-sizes-Linux-(){ cat << EOS
    -1
    -3
    -10
    -EOS
    -}
    -cudarap-prepare-sizes-Darwin-(){ cat << EOS
    -1
    -3
    -EOS
    -}
    -cudarap-prepare-sizes(){ $FUNCNAME-$(uname)- ; }
    +cudarap-prepare-sizes-Linux-(){  echo ${OPTICKS_CUDARAP_RNGMAX:-1,3,10} ; }
    +cudarap-prepare-sizes-Darwin-(){ echo ${OPTICKS_CUDARAP_RNGMAX:-1,3} ; }
    +cudarap-prepare-sizes(){ $FUNCNAME-$(uname)- | tr "," "\n"  ; }
     




