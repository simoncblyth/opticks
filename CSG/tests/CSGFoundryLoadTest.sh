#!/bin/bash

usage(){ cat << EOU
CSGFoundryLoadTest.sh
========================

~/o/CSG/tests/CSGFoundryLoadTest.sh

LVID=32 TEST=getMeshPrim ~/o/CSG/tests/CSGFoundryLoadTest.sh




EOU

}

cd $(dirname $(realpath $BASH_SOURCE))

name=CSGFoundryLoadTest
bin=$name
script=$name.py


#test=Load
#test=getMeshPrim
#test=descPrimRange
test=CompareRanges

export TEST=${TEST:-$test}
source $HOME/.opticks/GEOM/GEOM.sh

vv="BASH_SOURCE name bin script PWD GEOM TEST LVID"

export SSim__load_tree_load=1

logging()
{
    type $FUNCNAME
    export CSGFoundry=INFO
    export SSim=INFO
}
[ -n "$LOG" ] && logging

defarg="info_run"
[ -n "$BP" ] && defarg="dbg"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi

exit 0


