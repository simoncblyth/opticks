#!/usr/bin/env bash
usage(){ cat << EOU
srpn_test.sh
=======================

~/o/sysrap/tests/srpn_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=srpn_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="info_gcc_run"
arg=${1:-$defarg}


vv="BASH_SOURCE PWD name tmp TMP FOLD bin defarg arg"

if [[ "$arg" =~ info ]]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ gcc ]]; then
   gcc $name.cc -std=c++17 -lstdc++ -lm -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - gcc ERROR && exit 1
fi

if [[ "$arg" =~ run ]]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run ERROR && exit 2
fi
exit 0

