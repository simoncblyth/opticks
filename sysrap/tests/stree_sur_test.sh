#!/bin/bash -l 
usage(){ cat << EOU
stree_sur_test.sh 
=====================

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

defarg="info_ana"
arg=${1:-$defarg}

name=stree_sur_test 

source $HOME/.opticks/GEOM/GEOM.sh 
export BASE=$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim
export FOLD=$BASE


if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE BASE FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $SDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 


