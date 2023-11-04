#!/bin/bash -l 

vars="BASH_SOURCE GEOM bin"
bin=U4Material_MakePropertyFold_LoadTest
source $HOME/.opticks/GEOM/GEOM.sh 

defarg=info_run
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi







