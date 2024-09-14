#!/bin/bash
usage(){ cat << EOU
S4MaterialPropertyVector_test.sh
==================================

::

    ~/o/sysrap/tests/S4MaterialPropertyVector_test.sh


        -I$(clhep-prefix)/include \
        -I$(g4-prefix)/include/Geant4  \
        -L$(clhep-prefix)/lib \


EOU
}



name=S4MaterialPropertyVector_test
cd $(dirname $(realpath $BASH_SOURCE))

get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }
GEANT4_PREFIX=$(get-cmake-prefix Geant4)
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)


defarg="info_build_run"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name


export TEST=VV_CombinedArray

vars="BASH_SOURCE name PWD defarg arg FOLD bin GEANT4_PREFIX CLHEP_PREFIX TEST"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
        -std=c++17 -lstdc++ \
        -I.. \
        -I${CLHEP_PREFIX}/include \
        -I${GEANT4_PREFIX}/include/Geant4 \
        -L${GEANT4_PREFIX}/lib64 \
        -lG4global \
        -L${CLHEP_PREFIX}/lib \
        -lCLHEP \
        -o $bin 

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi

exit 0 


