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

defarg="info_nvcc_gcc_run"
arg=${1:-$defarg}
export FOLD=/tmp/$name
mkdir -p $FOLD

cuo=$FOLD/QScintThree_cu.o
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


# ordinary "lo" environment is sufficient
get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)
GEANT4_PREFIX=$(get-cmake-prefix Geant4)


vars="BASH_SOURCE PWD name defarg arg FOLD GEOM CUDA_PREFIX CLHEP_PREFIX GEANT4_PREFIX"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/nvcc}" != "$arg" ]; then
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
   gcc \
       $name.cc \
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
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

exit 0

