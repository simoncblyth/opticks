#!/bin/bash

usage(){ cat << EOU
QScintThree_test.sh
=====================

~/o/qudarap/tests/QScintThree_test.sh


EOU
}

name=QScintThree_test

source $HOME/.opticks/GEOM/GEOM.sh

cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_nvcc_gcc_run_pdb"
arg=${1:-$defarg}

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

cuo=$FOLD/QScintThree_cu.o
bin=$FOLD/$name
script=$name.py

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


# ordinary "lo" environment is sufficient
get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)
GEANT4_PREFIX=$(get-cmake-prefix Geant4)

#spec=M10
spec=M100

export U4ScintThree__num_wlsamp=$spec
export Q4ScintThree__num_wlsamp=$spec

unset Q4ScintThree__SD
export Q4ScintThree__SD=1   
if [ -n "$Q4ScintThree__SD" ]; then
   printf " ===== WARNING HD20 HAS BEEN DISABLED VIA Q4ScintThree__SD \n"
fi




vars="BASH_SOURCE PWD name defarg arg tmp TMP FOLD cuo bin GEOM CUDA_PREFIX CLHEP_PREFIX GEANT4_PREFIX U4ScintThree__num_wlsamp Q4ScintThree__num_wlsamp"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/nvcc}" != "$arg" ]; then
   echo nvcc
   nvcc \
       -c \
       ../QScintThree.cu \
       -I.. \
       -I../../sysrap \
       -std=c++17 -lstdc++  \
       -o $cuo
   [ $? -ne 0 ] && echo $BASH_SOURCE : nvcc error && exit 1
fi

if [ "${arg/gcc}" != "$arg" ]; then
   echo gcc
   gcc \
       $name.cc \
       $cuo \
       -std=c++17 -lstdc++ -lcudart -g \
       -DWITH_CUDA \
       -DRNG_PHILOX \
       -I.. \
       -I../../sysrap \
       -I../../u4 \
       -I$CUDA_PREFIX/include \
       -I$CLHEP_PREFIX/include \
       -I$GEANT4_PREFIX/include/Geant4  \
       -L$GEANT4_PREFIX/lib64 \
       -L$CLHEP_PREFIX/lib \
       -lG4global \
       -lG4geometry \
       -lCLHEP \
       -L$CUDA_PREFIX/lib64 \
       -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : gcc error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
   mode=-2
   MODE=${MODE:-$mode} ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

exit 0

