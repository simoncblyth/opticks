#!/bin/bash
usage(){ cat << EOU
SRecord_test.sh
======================

~/o/sysrap/tests/SRecord_test.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SRecord_test
tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

source $HOME/.opticks/GEOM/GEOM.sh
source $HOME/.opticks/GEOM/EVT.sh

export SRECORD_FOLD=/tmp/sphoton_test/record.npy

vars="BASH_SOURCE PWD FOLD SRECORD_FOLD bin AFOLD AFOLD_RECORD_SLICE"

defarg=info_build_run
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

glm_prefix=$OPTICKS_PREFIX/externals/glm/glm
GLM_PREFIX=${GLM_PREFIX:-$glm_prefix}



if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then

    echo $BASH_SOURCE [ building
    gcc  $name.cc \
         -std=c++17 -lstdc++ -lm \
         -I.. -g \
         -I$OPTICKS_PREFIX/externals/glm/glm \
         -I$OPTICKS_PREFIX/externals/include \
         -I$OPTICKS_PREFIX/include/SysRap \
         -I$CUDA_PREFIX/include \
         -lstdc++ \
         -lm \
         -I$GLM_PREFIX \
         -I${CUDA_PREFIX}/include \
         -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
    echo $BASH_SOURCE ] building

fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source  dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi

exit 0
