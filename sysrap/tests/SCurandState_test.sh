#!/bin/bash

usage(){ cat << EOU
SCurandState_test.sh
======================

~/o/sysrap/tests/SCurandState_test.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=SCurandState_test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

#test=NumFromFilesize
#test=ParseDir
#test=ChunkLoadSave
#test=ctor
test=load

export TEST=${TEST:-$test}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


defarg="info_build_run"
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg name FOLD bin test TEST CUDA_PREFIX"





gdb__ () 
{ 
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi 

if [ "${arg/build}" != "$arg" ]; then
 gcc $name.cc -std=c++11 -lstdc++ -g -I.. -I$CUDA_PREFIX/include -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi

if [ "${arg/prep}" != "$arg" ]; then
   iwd=$PWD 
   cd $HOME/.opticks/rngcache/RNG
   touch SCurandChunk_0000_1M_0_0.bin
   touch SCurandChunk_0001_1M_0_0.bin
   touch SCurandChunk_0002_1M_0_0.bin
   touch SCurandChunk_0003_1M_0_0.bin
   cd $iwd
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    gdb__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi


exit 0 
