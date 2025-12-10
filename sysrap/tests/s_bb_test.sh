#!/bin/bash
usage(){ cat << EOU
s_bb_test.sh
=============

~/o/sysrap/tests/s_bb_test.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

name=s_bb_test
export FOLD=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="info_build_run"
arg=${1:-$defarg}

vars="BASH_SOURCE PWD defarg arg name FOLD bin TEST"

#test=ALL
test=Degenerate
export TEST=${TEST:-$test}

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc \
         $name.cc \
         ../s_bb.cc \
         -g \
         -std=c++17 -lstdc++ \
         -I.. \
         -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

exit 0

