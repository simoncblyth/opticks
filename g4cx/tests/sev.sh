#!/bin/bash -l 
usage(){ cat << EOU
sev.sh
=========

Update geometry, transfer to laptop, load into ipython::

    GEOM get                         
    ~/opticks/g4cx/tests/sev.sh 

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
script=$SDIR/sev.py 
source $HOME/.opticks/GEOM/GEOM.sh 
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/ntds3
export FOLD=$BASE/ALL1/p001

defarg="info_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE GEOM BASE FOLD script" 

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi

