#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/scurand_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=scurand_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

defarg="info_gcc_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name tmp TMP FOLD bin PWD defarg arg"

if [[ "$arg" =~ info ]]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ gcc ]]; then
    gcc $name.cc -g -I.. -DMOCK_CURAND -std=c++17 -lstdc++ -lm -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE - gcc ERROR && exit 1
fi

if [[ "$arg" =~ run ]]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE - run ERROR && exit 1
fi

exit 0

