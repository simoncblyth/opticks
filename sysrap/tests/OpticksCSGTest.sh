#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/OpticksCSGTest.sh
TEST=TypeCodeVec ~/o/sysrap/tests/OpticksCSGTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=OpticksCSGTest

FOLD=/tmp/$USER/opticks/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="info_build_run"
arg=${1:-$defarg}

vv="PWD name defarg arg FOLD bin"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++17 -lstdc++ -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

exit 0


