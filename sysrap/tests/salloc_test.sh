#!/bin/bash
usage(){ cat << EOU
salloc_test.sh
===============

~/o/sysrap/tests/salloc_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh

defarg="info_build_run"
arg=${1:-$defarg}

name=salloc_test

source $HOME/.opticks/GEOM/GEOM.sh
export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py

export BASE=${BASE:-$FOLD}

vars="BASH_SOURCE name FOLD BASE OPTICKS_PREFIX GEOM TEST"


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi

exit 0

