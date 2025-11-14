#!/bin/bash
usage(){ cat << EOU
SPM_test.sh
===========

~/o/sysrap/tests/SPM_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SPM_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

afold=/tmp/blyth/opticks/GEOM/J25_4_0_opticks_Debug/python3.11/ALL0_no_opticks_event_name/A000
export AFOLD=${AFOLD:-$afold}

#test=merge_partial_select
test=dump_partial

export TEST=${TEST:-$test}


mkdir -p $FOLD

defarg="info_nvcc_gcc_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name tmp TMP FOLD bin defarg arg cuda_prefix CUDA_PREFIX test TEST AFOLD"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/ls}" != "$arg" ]; then
   ff="AFOLD FOLD"
   for f in $ff ; do
      cmd="ls -alst ${!f} # $f"
      echo $cmd
      eval $cmd
   done
fi


if [ "${arg/nvcc}" != "$arg" ]; then
   nvcc -c ../SU.cu -I.. -std=c++17 --extended-lambda  -o $FOLD/SU.o
   [ $? -ne 0 ] && echo $FUNCNAME nvcc compile error && exit 1
fi

if [ "${arg/nvcc}" != "$arg" ]; then
   nvcc -c ../SPM.cu -I.. -std=c++17 --extended-lambda  -o $FOLD/SPM.o
   [ $? -ne 0 ] && echo $FUNCNAME nvcc compile error && exit 1
fi

if [ "${arg/gcc}" != "$arg" ]; then
   gcc $name.cc $FOLD/SU.o $FOLD/SPM.o -std=c++17 -lstdc++ -I.. -I$CUDA_PREFIX/include -L$CUDA_PREFIX/lib64 -lcudart -lm  -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE gcc compile error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
   pushd $FOLD
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 3
   popd
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   pushd $FOLD
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 4
   popd
fi

exit 0

