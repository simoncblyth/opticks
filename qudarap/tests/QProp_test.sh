#!/bin/bash
usage(){ cat << EOU
QProp_test.sh
===============

~/o/qudarap/tests/QProp_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=QProp_test

source $HOME/.opticks/GEOM/GEOM.sh

defarg="info_build_run_ana"
arg=${1:-$defarg}


tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py

vv="BASH_SOURCE tmp TMP FOLD name GEOM ${GEOM}_CFBaseFromGEOM bin script"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%20s : %s \n" "$v" "${!v}" ; done
fi


if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc ../QProp.cc  \
       -g -std=c++17 -lstdc++ -lm \
       -DMOCK_CURAND \
       -I.. \
       -I../../sysrap \
       -I$CUDA_PREFIX/include \
       -I$OPTICKS_PREFIX/externals/glm/glm \
       -I$OPTICKS_PREFIX/externals/plog/include \
       -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
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

