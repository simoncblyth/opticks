#!/usr/bin/env bash

usage(){ cat << EOU


~/o/sysrap/tests/SGLM_View_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SGLM_View_test
prepscript=$name.py

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

defarg="info_prep_build_run"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name prepscript tmp TMP FOLD bin defarg arg"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/prep}" != "$arg" ]; then
    ${PYTHON:-python} $prepscript
    [ $? -ne 0 ] && echo $BASH_SOURCE prep error && exit 1
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++17 -lstdc++ -lm -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE gcc error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 3
fi

exit 0

