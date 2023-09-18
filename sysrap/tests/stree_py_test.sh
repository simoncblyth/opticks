#!/bin/bash -l 

cd $(dirname $BASH_SOURCE)

name=stree_py_test
defarg="info_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE GEOM"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1
fi

exit 0 
