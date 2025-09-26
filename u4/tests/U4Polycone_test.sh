#!/bin/bash
usage(){ cat << EOU
U4Polycone_test.sh
===================

This compiles the needed parts of sysrap from source (not using libSysRap)
in order to facilitate quick changes of compilation options.

Compilation options:

former -DWITH..SND
     reverted to the old inflexible snd.hh CSG node impl
     instead of the default more flexible sn.h impl

-DWITH_CHILD
     switches sn.h node impl to use child vector instead of left right nodes,
     this generalizes the node impl to n-ary instead of binary

EOU
}

opt="-DWITH_CHILD"


SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

name=U4Polycone_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$SDIR/$name.py

CUDA_PREFIX=/usr/local/cuda

defarg="info_build_run_pdb"
arg=${1:-$defarg}

export sn__level=2
export s_pool_level=2

get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)
GEANT4_PREFIX=$(get-cmake-prefix Geant4)

vars="BASH_SOURCE arg SDIR FOLD bin script opt"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then

    gcc \
         $opt \
         $SDIR/$name.cc \
         $SDIR/../../sysrap/sn.cc \
         $SDIR/../../sysrap/s_pa.cc \
         $SDIR/../../sysrap/s_bb.cc \
         $SDIR/../../sysrap/s_tv.cc \
         $SDIR/../../sysrap/s_csg.cc \
         \
         $SDIR/../../sysrap/snd.cc \
         $SDIR/../../sysrap/scsg.cc \
         -I$SDIR/.. \
         -std=c++17 -lstdc++ -g \
         -I$HOME/opticks/sysrap \
         -I$CUDA_PREFIX/include \
         -I$OPTICKS_PREFIX/externals/glm/glm \
         -I$CLHEP_PREFIX/include \
         -I$GEANT4_PREFIX/include/Geant4  \
         -L$GEANT4_PREFIX/lib64 \
         -L$CLHEP_PREFIX/lib \
         -lG4global \
         -lG4geometry \
         -lG4graphics_reps \
         -lCLHEP \
         -lm \
         -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi

exit 0


