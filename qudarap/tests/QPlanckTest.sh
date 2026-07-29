#!/bin/bash

usage(){ cat << EOU

~/o/qudarap/tests/QPlanckTest.sh

EOU
}

name=QPlanckTest
bin=$name
script=$name.py

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export FOLD=$TMP/$name
mkdir -p $FOLD

defarg="info_run_ls"
arg=${1:-$defarg}

cd $(dirname $(realpath $BASH_SOURCE))

vv="BASH_SOURCE name bin tmp TMP FOLD PWD defarg arg"

if [[ "$arg" =~ info ]]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ run ]]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run ERROR && exit 1
fi

if [[ "$arg" =~ ls ]]; then
   ls -alst $FOLD
fi

if [[ "$arg" =~ "pdb" ]]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE - pdb ERROR for script $script && exit 1
fi



