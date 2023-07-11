#!/bin/bash -l 


name=CSGFoundry_py_test
source $HOME/.opticks/GEOM/GEOM.sh 

defarg="info_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE name arg GEOM"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi 

exit 0 
