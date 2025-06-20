#!/bin/bash
usage(){ cat << EOU
sseq_record_test.sh
======================

::

    ~/opticks/sysrap/tests/sseq_record_test.sh info


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

source $HOME/.opticks/GEOM/GEOM.sh
source $HOME/.opticks/GEOM/EVT.sh

TMP=${TMP:-/tmp/$USER/opticks}

name=sseq_record_test
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name


defarg="info_build_run"
arg=${1:-$defarg}

vars=""
vars="$vars BASH_SOURCE GEOM AFOLD AFOLD_RECORD_SLICE BFOLD BFOLD_RECORD_SLICE name bin"


if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++11 -lstdc++ -I.. -I$CUDA_PREFIX/include -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi


exit 0


