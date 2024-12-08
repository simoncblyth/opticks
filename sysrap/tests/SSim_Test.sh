#!/bin/bash
usage(){ cat << EOU
SSim_Test.sh
=============

~/o/sysrap/tests/SSim_Test.sh


EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=SSim_Test
script=$name.py

source $HOME/.opticks/GEOM/GEOM.sh
export FOLD=$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim

#defarg="info_ana"
defarg="info_pdb"
arg=${1:-$defarg}

vars="BASH_SOURCE name script GEOM FOLD HOME defarg arg"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python}  $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 



