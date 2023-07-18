#!/bin/bash -l 
usage(){ cat << EOU
QBndTest.sh
============

EOU
}

name=QBndTest
defarg="info_run_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh 
export FOLD=/tmp/$USER/opticks/$name

vars="BASH_SOURCE name arg GEOM FOLD"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 2 
fi 

exit 0 

