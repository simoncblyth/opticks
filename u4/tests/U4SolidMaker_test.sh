#!/bin/bash
usage(){ cat << EOU


~/o/u4/tests/U4SolidMaker_test.sh

FAILS to compile due to SLOG, so revert to U4SolidMakerTest.cc

EOU
}

name=U4SolidMaker_test

cd $(dirname $(realpath $BASH_SOURCE))

export FOLD=/tmp/$USER/opticks/$name
mkdir -p $FOLD

get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)
G4_PREFIX=$(get-cmake-prefix Geant4)

bin=$FOLD/$name

opt=-DWITH_CHILD


vv="BASH_SOURCE name PWD bin FOLD arg defarg CLHEP_PREFIX G4_PREFIX"

defarg="info_build_run"
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then

    gcc \
    $opt \
    $name.cc \
    ../U4SolidMaker.cc \
    ../../sysrap/sn.cc \
    ../../sysrap/s_pa.cc \
    ../../sysrap/s_bb.cc \
    ../../sysrap/s_tv.cc \
    ../../sysrap/s_csg.cc \
    \
    ../../sysrap/snd.cc \
    ../../sysrap/scsg.cc \
    -I.. \
    -std=c++17 -lstdc++ \
    -I../../sysrap \
    -I$CUDA_PREFIX/include \
    -I$OPTICKS_PREFIX/externals/glm/glm \
    -I$OPTICKS_PREFIX/externals/plog/include \
    -I$CLHEP_PREFIX/include \
    -I$G4_PREFIX/include/Geant4  \
    -L$G4_PREFIX/lib64 \
    -L$CLHEP_PREFIX/lib \
    -lG4global \
    -lG4geometry \
    -lG4graphics_reps \
    -lCLHEP \
    -lm \
    -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE - build error && exit 1

fi


