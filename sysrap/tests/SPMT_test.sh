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
to SPMT::CreateFromJPMT the PMT info NPFold from SPMT::PATH::

    $CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source $HOME/.opticks/GEOM/GEOM.sh

#defarg="info_build_run_ana"
defarg="info_build_run_pdb"
arg=${1:-$defarg}

name=SPMT_test
script=$name.py

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name

mkdir -p $FOLD
bin=$FOLD/$name

#test=ART
test=testfold

export TEST=${TEST:-$test}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

unset SPMT__level
#export SPMT__level=1


vars="BASH_SOURCE BASH_VERSION defarg arg name GEOM TMP FOLD CUDA_PREFIX bin script TEST SPMT__level"

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
    source dbg__.sh
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


