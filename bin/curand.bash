curand-source(){ echo $BASH_SOURCE ; }
curand-vi(){ vi $(curand-source)  ; }
curand-env(){  olocal- ; opticks- ; }
curand-usage(){ cat << EOU

curand
==================================================================================================


skipahead
------------

* https://docs.nvidia.com/cuda/curand/device-api-overview.html#skip-ahead


registers per thread limits
------------------------------

* https://stackoverflow.com/questions/19737812/curand-device-api-and-seed




EOU
}



