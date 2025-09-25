#!/usr/bin/env bash

usage(){ cat << EOU

~/o/u4/tests/StrutMakerTest.sh

EOU
}

name=StrutMakerTest
script=U4Mesh_test.py

cd $(dirname $(realpath $BASH_SOURCE))

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name


get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)
GEANT4_PREFIX=$(get-cmake-prefix Geant4)

vv="BASH_SOURCE name script bin FOLD PWD CLHEP_PREFIX GEANT4_PREFIX defarg arg SOL SOLID TITLE"

defarg="info_build_run_pdb"
arg=${1:-$defarg}


prefix="StrutAcrylicConstruction StrutBar2AcrylicConstruction"
suffix="_Complex _Simple _Complex_VERBOSE _Simple_VERBOSE"



if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
         -I.. \
         -std=c++17 -lstdc++ \
         -I$HOME/junosw/Simulation/DetSimV2/CentralDetector/include \
         -I$HOME/opticks/sysrap \
         -I$CLHEP_PREFIX/include \
         -I$GEANT4_PREFIX/include/Geant4  \
         -L$GEANT4_PREFIX/lib64 \
         -L$CLHEP_PREFIX/lib \
         -lG4global \
         -lG4geometry \
         -lG4graphics_reps \
         -lCLHEP \
         -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    for pfx in $prefix ; do
        for sfx in $suffix ; do
             export SOLID=${pfx}${sfx}
             export TITLE="$BASH_SOURCE : $script : $SOLID"
             $bin
             [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
        done
    done
fi

if [ "${arg/dbg}" != "$arg" ]; then
    export SOLID=StrutAcrylicConstruction_Complex
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi

if [ "${arg/pdb}" != "$arg" ]; then
    for pfx in $prefix ; do
        for sfx in $suffix ; do
             export SOLID=${pfx}${sfx}
             export TITLE="$BASH_SOURCE : $script : $SOLID"

             ${IPYTHON:-ipython} --pdb -i $script
             [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 4

        done
    done
fi

exit 0


