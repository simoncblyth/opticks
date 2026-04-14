#!/usr/bin/env bash

usage(){ cat << EOU

~/o/qudarap/tests/QTexLookupTest.sh

EOU
}

defarg=info_run_ls_pdb
arg=${1:-$defarg}

cd $(dirname $(realpath $BASH_SOURCE))

name=QTexLookupTest
bin=$name
script=$name.py
export FOLD=$TMP/$name

vv="BASH_SOURCE PWD defarg arg name bin FOLD script"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - run fail && exit 1
fi

if [ "${arg/ls}" != "$arg" ]; then
    echo ls -alst $FOLD
    ls -alst $FOLD
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - ls fail && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - pdb fail && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - ana fail && exit 4
fi
exit 0


