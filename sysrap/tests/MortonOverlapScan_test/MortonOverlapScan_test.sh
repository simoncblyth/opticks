#!/bin/bash
usage(){ cat << EOU

MortonOverlapScan_test.sh
==========================

::

    ~/o/sysrap/tests/MortonOverlapScan_test/MortonOverlapScan_test.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=MortonOverlapScan_test
script=$name.py

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

mfold=/data1/blyth/tmp/GEOM/J25_7_2_opticks_Debug/CSGOptiXTMTest/ALL0_2dxz_tall_wide_radial_range/A000
export MFOLD=${MFOLD:-$mfold}
simtrace="$MFOLD/simtrace.npy"

if [ ! -f "$simtrace" ]; then
    echo $BASH_SOURCE - FATAL - simtrace $simtrace DOES NOT EXIST - use cx:cxt_min.sh to create simtrace.npy
    exit 1
fi

test=overlap
unset TEST
export TEST=${TEST:-$test}


defarg="info_nvcc_gcc_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name tmp TMP FOLD bin script defarg arg cuda_prefix CUDA_PREFIX test TEST mfold MFOLD"

log(){ echo $BASH_SOURCE $(date +%H:%M:%S) $* ; }


if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/ls}" != "$arg" ]; then
   ff="MFOLD FOLD"
   for f in $ff ; do
      cmd="ls -alst ${!f} # $f"
      echo $cmd
      eval $cmd
   done
fi

if [ "${arg/nvcc}" != "$arg" ]; then
   log nvcc ../../SU.cu
   nvcc -c ../../SU.cu -I../.. -std=c++17 --extended-lambda  -o $FOLD/SU.o
   [ $? -ne 0 ] && echo $FUNCNAME nvcc compile error && exit 1
fi

if [ "${arg/nvcc}" != "$arg" ]; then
   log nvcc $name.cu
   nvcc -c MortonOverlapScan.cu -I../.. -std=c++17 --extended-lambda  -o $FOLD/$name.cu.o
   [ $? -ne 0 ] && echo $FUNCNAME nvcc compile error && exit 1
fi

if [ "${arg/gcc}" != "$arg" ]; then
   log gcc $name.cc
   gcc $name.cc $FOLD/SU.o $FOLD/$name.cu.o  -std=c++17 -lstdc++ -DWITH_CUDA -I../.. -I$CUDA_PREFIX/include -L$CUDA_PREFIX/lib64 -lcudart -lm  -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE gcc compile error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
   log run
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 3
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 4
fi

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 5
fi


exit 0

