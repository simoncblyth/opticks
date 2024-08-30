#!/bin/bash -l 
usage(){ cat << EOU
sdevice_test.sh
=================

This assumes that the ordinal is the index when all GPUs are visible 
and it finds this by arranging to persist the query when 
CUDA_VISIBLE_DEVICES is not defined and use that to provide something 
to match against when the envvar is defined.

The purpose is for reference running, especially performance
scanning : so its acceptable to require running a metadata
capturing executable prior to scanning.
That initial executable can be this one and the 
ones with CUDA_VISIBLE_DEVICES can be embedded opticks
running. 

The typical usage would be to write GPU description
into run/event metadata. Or could access the sdevice.h struct 
to put more details such as the VRAM into run/event metadata.  


Initial run without CUDA_VISIBLE_DEVICES envvar defined
writes info about all connected GPUs to "$HOME/.opticks/runcache"::

    ~/opticks/sysrap/tests/sdevice_test.sh build_run

Subsequent runs with CUDA_VISIBLE_DEVICES match the currently 
visible GPUs against all of them without restriction, so 
original ordinal can be discerned even when running with 
a subset of the GPUs::

    CUDA_VISIBLE_DEVICES=1   ~/opticks/sysrap/tests/sdevice_test.sh run 
    CUDA_VISIBLE_DEVICES=0   ~/opticks/sysrap/tests/sdevice_test.sh run 
    CUDA_VISIBLE_DEVICES=0,1 ~/opticks/sysrap/tests/sdevice_test.sh run 
    CUDA_VISIBLE_DEVICES=1,0 ~/opticks/sysrap/tests/sdevice_test.sh run 


HMM : WHAT ABOUT IDENTICAL GPUs ?
a hidden uuid is used in the matching so should work. 


Examples::

    [blyth@localhost ~]$ ~/opticks/sysrap/tests/sdevice_test.sh 
    [0:NVIDIA_RTX_5000_Ada_Generation]
    idx/ord/mpc/cc:0/0/100/89  31.592 GB  NVIDIA RTX 5000 Ada Generation

    N[blyth@localhost ~]$ ~/opticks/sysrap/tests/sdevice_test.sh build_run
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX

    N[blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=0 ~/opticks/sysrap/tests/sdevice_test.sh run
    [0:TITAN_V]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V

    N[blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=1 ~/opticks/sysrap/tests/sdevice_test.sh run
    [1:TITAN_RTX]
    idx/ord/mpc/cc:0/1/72/75  23.652 GB  TITAN RTX

    N[blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=1,0 ~/opticks/sysrap/tests/sdevice_test.sh run
    [1:TITAN_RTX 0:TITAN_V]
    idx/ord/mpc/cc:0/1/72/75  23.652 GB  TITAN RTX
    idx/ord/mpc/cc:1/0/80/70  11.784 GB  TITAN V

    N[blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=0,1 ~/opticks/sysrap/tests/sdevice_test.sh run
    [0:TITAN_V 1:TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX


EOU
}

cd $(dirname $BASH_SOURCE)
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


