#!/bin/bash
usage(){ cat << EOU
G4CXOpticks_setGeometry_Default_Test.sh
========================================


EOU
}

cd $(dirname $BASH_SOURCE)

defarg=info_dbg_ana
arg=${1:-$defarg}
bin=G4CXOpticks_setGeometry_Test

source $HOME/.opticks/GEOM/GEOM.sh   # mini config script that only sets GEOM envvar

vars="BASH_SOURCE arg GEOM bin"


logging(){
   export Dummy=INFO
   export G4CXOpticks=INFO
   export X4PhysicalVolume=INFO
   #export SOpticksResource=INFO
   export CSGFoundry=INFO
   export GSurfaceLib=INFO
   export U4VolumeMaker=INFO
   #export NCSG=INFO
}
[ -n "$LOG" ] && logging && env | grep =INFO


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    export TAIL="-o run"
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

#if [ "${arg/ana}" != "$arg" ]; then
#    ${IPYTHON:-ipython} --pdb -i $script
#    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
#fi

exit 0

