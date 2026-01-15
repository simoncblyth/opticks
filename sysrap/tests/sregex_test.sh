#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/sregex_test.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

name=sregex_test

FOLD=/tmp/$USER/opticks/$name
mkdir -p $FOLD

bin=$FOLD/$name

defarg="info_build_run"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name FOLD bin defarg arg REGEX"

export REGEX='s_EMFcoil_holder_ring(_mod)?[0-9]+_seg[0-9]+'  # NB single quote protection from bash

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++17 -lstdc++ -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build ERROR && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run ERROR && exit 2
fi

exit 0


