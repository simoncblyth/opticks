#!/usr/bin/env bash

usage(){ cat << EOU
opticks_CSGOptiX_test.sh
==========================

~/o/CSGOptiX/tests/opticks_CSGOptiX_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

source $HOME/.opticks/GEOM/GEOM.sh

defarg="info_run"
arg=${1:-$defarg}

name=opticks_CSGOptiX_test
script=$name.py

vv="BASH_SOURCE PWD name script GEOM"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} -i --pdb  $script
    [ $? -ne 0 ] && echo $BASH_SOURCE - pdb error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $(which python) $PWD/$script
    [ $? -ne 0 ] && echo $BASH_SOURCE - dbg error && exit 2
fi

exit 0

