#!/bin/bash -l 
usage(){ cat << EOU
scontext_test.sh
=================

* note that the nvidia-smi ordering does not necessarily match the CUDA ordering 
* also I think that nvidia-smi ordering may change after reboots 
* also the CUDA ordering can be changed with eg CUDA_VISIBLE_DEVICES=1,0 


::


    N[blyth@localhost tests]$ ./scontext_test.sh 
    0:TITAN_V 1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=1 ./scontext_test.sh 
    1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=0 ./scontext_test.sh 
    0:TITAN_V
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=1 ./scontext_test.sh 
    1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=0,1 ./scontext_test.sh 
    0:TITAN_V 1:TITAN_RTX
    N[blyth@localhost tests]$ CUDA_VISIBLE_DEVICES=1,0 ./scontext_test.sh 
    1:TITAN_RTX 0:TITAN_V
    N[blyth@localhost tests]$ 



    N[blyth@localhost tests]$ VERBOSE=1 ./scontext_test.sh 
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX


    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=0 ./scontext_test.sh 
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [0:TITAN_V]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V

    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=1 ./scontext_test.sh 
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [1:TITAN_RTX]
    idx/ord/mpc/cc:0/1/72/75  23.652 GB  TITAN RTX

    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=1,0 ./scontext_test.sh 
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [1:TITAN_RTX 0:TITAN_V]
    idx/ord/mpc/cc:0/1/72/75  23.652 GB  TITAN RTX
    idx/ord/mpc/cc:1/0/80/70  11.784 GB  TITAN V

    N[blyth@localhost tests]$ VERBOSE=1 CUDA_VISIBLE_DEVICES=0,1 ./scontext_test.sh 
    scontext::desc
    all_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX
    visible_devices
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX



    N[blyth@localhost tests]$ nvidia-smi
    Mon Jun  5 19:51:47 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0 Off |                  N/A |
    | 39%   55C    P0    71W / 280W |      0MiB / 24219MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:A6:00.0 Off |                  N/A |
    | 46%   54C    P8    N/A /  N/A |      0MiB / 12066MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+



EOU
}

name=scontext_test
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

defarg="build_run"
arg=${1:-$defarg}

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

CUDA_LIBDIR=$CUDA_PREFIX/lib
[ ! -d "$CUDA_LIBDIR" ] && CUDA_LIBDIR=$CUDA_PREFIX/lib64


if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc  \
       -g -std=c++11 -lstdc++ \
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


