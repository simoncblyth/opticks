#!/bin/bash
usage(){ cat << EOU
SPMT_test.sh
===============

::

    ~/opticks/sysrap/tests/SPMT_test.sh


For AOI scanning use::

    ./SPMT_scan.sh

That does something like::

    N_MCT=900 N_SPOL=1  ./SPMT_test.sh

Note dependency on GEOM envvar, which is used
to SPMT::Load the PMT info NPFold from::

    $HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/extras/jpmt

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source $HOME/.opticks/GEOM/GEOM.sh

defarg="info_build_run_ana"
arg=${1:-$defarg}

name=SPMT_test
script=$name.py

export TMP=${TMP:-/tmp/$USER/opticks}
export SFOLD=$TMP/$name
export JFOLD=$TMP/JPMTTest

mkdir -p $SFOLD
bin=$SFOLD/$name

#test=ART
test=testfold

export TEST=${TEST:-$test}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="arg name REALDIR GEOM TMP FOLD SFOLD JFOLD CUDA_PREFIX bin script TEST"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$HOME/customgeant4 \
           -DWITH_CUSTOM4 \
           -g -std=c++17 -lstdc++ -lm \
           -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
    echo $BASH_SOURCE : build OK
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 4
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 5
fi

exit 0


