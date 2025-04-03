#!/bin/bash

usage(){ cat << EOU

~/o/CSG/tests/CSGFoundryTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source ../../bin/dbg__.sh


logging()
{
    type $FUNCNAME
    export CSGFoundry=INFO
}

[ -n "$LOG" ] && logging

bin=CSGFoundryTest
defarg="info_run"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
export TEST=Load_Save
#export TEST=getPrimName
export CSGFoundry__SAVE_OPT=meshname

if [ "${arg/info}" != "$arg" ]; then
   vv="0 defarg arg bin GEOM ${GEOM}_CFBaseFromGEOM TEST CSGFoundry__SAVE_OPT"
   for v in $vv ; do printf "%20s : %s \n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $0 - run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $bin
   [ $? -ne 0 ] && echo $0 - dbg error && exit 2
fi

exit 0
