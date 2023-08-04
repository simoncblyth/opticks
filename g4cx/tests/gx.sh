#!/bin/bash -l 
usage(){ cat << EOU
gx.sh : python examination of persisted CSGFoundry and stree geometry
=========================================================================

After a run on workstation that persists into $HOME/.opticks/GEOM/$GEOM::

    GEOM       # laptop, check configured GEOM matches that on workstation
    GEOM get   # laptop, pullback geometry
    ~/opticks/g4cx/tests/gx.sh   # laptop, run this script

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
script=$SDIR/gx.py 
source $HOME/.opticks/GEOM/GEOM.sh   
vars="BASH_SOURCE SDIR GEOM script"

defarg="info_ana"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi
exit 0 

