#!/bin/bash -l 

name=CSGOptiXInstancesTest

defarg="info_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh 

#fold=/tmp/GEOM/$GEOM/CSGOptiX
fold=$HOME/.opticks/GEOM/$GEOM/CSGOptiX

export FOLD=$fold

vars="BASH_SOURCE name FOLD"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi
