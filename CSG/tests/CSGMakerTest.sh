#!/bin/bash
usage(){ cat << EOU

~/o/CSG/tests/CSGMakerTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

bin=CSGMakerTest

logging(){
   type $FUNCNAME
   export CSGFoundry=INFO
}
[ -n "$LOG" ] && logging


#geom=JustOrb
#geom=BoxedSphere
#export CSGMakerTest_GEOM=$geom


defarg=info_dbg
arg=${1:-$defarg}

vv="BASH_SOURCE PWD bin defarg arg geom CSGMakerTest_GEOM"

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

exit 0









