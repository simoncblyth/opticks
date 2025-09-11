#!/usr/bin/env bash


bin=CSGOptiXServiceTest
source $HOME/.opticks/GEOM/GEOM.sh

vars="BASH_SOURCE bin GEOM"

defarg="info_run"
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi


exit 0

