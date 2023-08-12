#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundryAB.sh 
================

Comparing two CSGFoundry. For example from:

1. standard G4CXOpticks::SetGeometry conversion  

   * $HOME/.opticks/GEOM/J007

2. experimental CSGFoundry saved after CSGFoundry::importTree from loaded stree

   * /tmp/$USER/opticks/CSGImportTest

EOU
}


SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
script=$SDIR/CSGFoundryAB.py

source $HOME/.opticks/GEOM/GEOM.sh 

a_cfbase=$HOME/.opticks/GEOM/$GEOM
b_cfbase=/tmp/GEOM/$GEOM
#b_cfbase=/tmp/$USER/opticks/CSGFoundry_CreateFromSimTest

export A_CFBASE=${A_CFBASE:-$a_cfbase}
export B_CFBASE=${B_CFBASE:-$b_cfbase}

defarg="info_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE SDIR GEOM A_CFBASE B_CFBASE arg script"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi 

exit 0 

