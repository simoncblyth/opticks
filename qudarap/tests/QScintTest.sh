#!/bin/bash

usage(){ cat << EOU

~/o/qudarap/tests/QScintTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=QScintTest

source $HOME/.opticks/GEOM/GEOM.sh

bin=$name
script=$name.py

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name

defarg="info_run_pdb"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name bin script defarg arg GEOM ${GEOM}_CFBaseFromGEOM tmp TMP FOLD"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
fi

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} -i --pdb $script
   [ $? -ne 0 ] && echo $BASH_SOURCE - pdb error && exit 1
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE - ana error && exit 2
fi

exit 0

