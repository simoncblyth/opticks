#!/bin/bash
usage(){ cat << EOU
curand_uniform_test.sh 
========================

~/o/sysrap/tests/curand_uniform_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh

name=curand_uniform_test
src=$name.cu
script=$name.py 

tmp=/data/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="info_build_run_ana"
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

OPT="-use_fast_math -DWITH_CURANDLITE"

M=1000000
ni=$(( 10*M ))
nj=16 
export NI=${NI:-$ni}
export NJ=${NJ:-$nj}

vars="BASH_SOURCE name src script bin FOLD OPT NI NJ"


if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    nvcc $src -std=c++17 $OPT -lcrypto -lssl -I$HOME/np -I..  -I$CUDA_PREFIX/include -o $bin
    [ $? -ne 0 ] && echo compilation error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo run  error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin
    [ $? -ne 0 ] && echo dbg  error && exit 2
fi 

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    python $script 
    [ $? -ne 0 ] && echo ana error && exit 4
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    rsync -av P:$FOLD/ $FOLD
    [ $? -ne 0 ] && echo grab error && exit 4
fi 



exit 0 

