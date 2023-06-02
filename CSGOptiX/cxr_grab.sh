#!/bin/bash -l 
usage(){ cat << EOU
cxr_grab.sh 
=============

::

   ./cxr_grab.sh grab 
   ./cxr_grab.sh open
   ./cxr_grab.sh clean 

Formerly used the below, but that hardcodes an old directory layout:: 

   EXECUTABLE=CSGOptiXRenderTest ./grab.sh $* 

For cxr_min.sh outputs use instead::

   ./cxr_min.sh grab_open  


EOU
}

DIR=$(dirname $BASH_SOURCE)
defarg="grab_open"
arg=${1:-$defarg}

cvd=0
CVD=${CVD:-$cvd}

source ~/.opticks/GEOM/GEOM.sh 

base=/tmp/$USER/opticks/CSGOptiX/$bin/SCVD$CVD/70000/-1/
export BASE=${BASE:-$base}
source $DIR/../bin/BASE_grab.sh $arg 

