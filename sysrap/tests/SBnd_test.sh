#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/SBnd_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SBnd_test

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py

source $HOME/.opticks/GEOM/GEOM.sh

defarg="info_build_run_pdb"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name tmp TMP FOLD bin script GEOM ${GEOM}_CFBaseFromGEOM defarg arg"

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -I.. -g -std=c++17 -lstdc++ -lcrypto -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 2
fi

exit 0

