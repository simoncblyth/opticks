#!/bin/bash -l 

name=QOpticalTest 
source $HOME/.opticks/GEOM/GEOM.sh 
vars="BASH_SOURCE name GEOM"

defarg="info_run"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi

exit 0

