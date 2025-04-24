#!/bin/bash
usage(){ cat << EOU
SSimTest.sh
=============

~/o/sysrap/tests/SSimTest.sh 
LOG=1 ~/o/sysrap/tests/SSimTest.sh dbg
TEST=addFake ~/o/sysrap/tests/SSimTest.sh dbg

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

source $HOME/.opticks/GEOM/GEOM.sh 
if [ -n "$UNSET" ]; then
     unset GEOM 
     echo $BASH_SOURCE check without GEOM
fi 

logging(){
   type $FUNCNAME
   export SSim=INFO
}
[ -n "$LOG" ] && logging 

name=SSimTest
defarg="info_run"
arg=${1:-$defarg}

vars="BASH_SOURCE name GEOM FOLD"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi

if [ "${arg/dbg}" != "$arg" ]; then 
   source dbg__.sh 
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

exit 0 
