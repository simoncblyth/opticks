#!/bin/bash
usage(){ cat << EOU
U4ScintThree_test.sh
======================

~/o/u4/tests/U4ScintThree_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=U4ScintThree_test
script=$name.py

defarg="info_gcc_run_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

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

if [ "${arg/gcc}" != "$arg" ]; then
   gcc \
       $name.cc \
       -std=c++17 -lstdc++ -g \
       -I../../sysrap \
       -I$CLHEP_PREFIX/include \
       -I$GEANT4_PREFIX/include/Geant4  \
       -I.. \
       -L$GEANT4_PREFIX/lib64 \
       -L$CLHEP_PREFIX/lib \
       -lG4global \
       -lG4geometry \
       -lCLHEP \
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

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

exit 0

