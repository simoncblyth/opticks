#!/bin/bash
usage(){ cat << EOU
SSimTest.sh
=============

~/o/sysrap/tests/SSimTest.sh 
TEST=addFake ~/o/sysrap/tests/SSimTest.sh dbg

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh 

source $HOME/.opticks/GEOM/GEOM.sh 
#unset GEOM # check without 

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
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

exit 0 
