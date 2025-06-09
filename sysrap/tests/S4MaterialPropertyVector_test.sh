#!/bin/bash
usage(){ cat << EOU
S4MaterialPropertyVector_test.sh
==================================

::

    ~/o/sysrap/tests/S4MaterialPropertyVector_test.sh


EOU
}


name=S4MaterialPropertyVector_test
cd $(dirname $(realpath $BASH_SOURCE))

get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }
GEANT4_PREFIX=$(get-cmake-prefix Geant4)
CLHEP_PREFIX=$(get-cmake-prefix CLHEP)


defarg="info_build_run_ana"
arg=${1:-$defarg}


tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py

#test=VV_CombinedArray
test=ConvertToArray
export TEST=${TEST:-$test}

vars="BASH_SOURCE name PWD defarg arg FOLD bin script GEANT4_PREFIX CLHEP_PREFIX test TEST"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
        -std=c++17 -lstdc++ -lm -g \
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

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi



if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi

exit 0


