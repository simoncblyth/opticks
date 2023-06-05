#!/bin/bash -l 
usage(){ cat << EOU
sdevice_test.sh
=================

::

    N[blyth@localhost tests]$ ./sdevice_test.sh
    cudaGetDeviceCount : 2
    sdevice::Visible no_cvd 
    path /home/blyth/.opticks/runcache/sdevice.bin
    visible devices[0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    N[blyth@localhost tests]$ 

    epsilon:tests blyth$ ./sdevice_test.sh 
    cudaGetDeviceCount : 1
    sdevice::Visible no_cvd 
    path /Users/blyth/.opticks/runcache/sdevice.bin
    visible devices[0:GeForce_GT_750M]
    idx/ord/mpc/cc:0/0/2/30   2.000 GB  GeForce GT 750M
    epsilon:tests blyth$ 

EOU
}

name=sdevice_test
bin=/tmp/$name

defarg="build_run"
arg=${1:-$defarg}

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

CUDA_LIBDIR=$CUDA_PREFIX/lib
[ ! -d "$CUDA_LIBDIR" ] && CUDA_LIBDIR=$CUDA_PREFIX/lib64


if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc  \
       -std=c++11 -lstdc++ \
       -I.. \
       -I$CUDA_PREFIX/include \
       -L$CUDA_LIBDIR \
       -lcudart \
       -o $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

exit 0 


