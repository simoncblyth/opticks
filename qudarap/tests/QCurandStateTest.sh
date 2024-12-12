#!/bin/bash 
usage(){ cat << EOU
QCurandStateTest.sh : testing the new chunk-centric curandState approach
=========================================================================

~/o/qudarap/tests/QCurandStateTest.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh 

name=QCurandStateTest

#defarg="info_dbg"
defarg="info_run"
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    gdb__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

exit 0 

