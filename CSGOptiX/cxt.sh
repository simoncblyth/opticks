#!/bin/bash -l 
msg="=== $BASH_SOURCE :"

catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && echo $msg catgeom $catgeom override of default geom $geom && geom=$(echo ${catgeom%%_*})
export GEOM=${GEOM:-$geom}


if [ "$(uname)" == "Linux" ]; then
    cfname=GeoChain/$GEOM            
else
    cfname=GeoChain_Darwin/$GEOM            
fi

export CFBASE=/tmp/$USER/opticks/$cfname
export TMIN=0
export EYE=-1,-1,-1
export CAMERATYPE=1

export FOLD=$CFBASE/CSGOptiX 
snap=$FOLD/snap.jpg

arg=${1:-run_ana}


if [ "${arg/dbg}" != "$arg" ]; then 
   lldb__ CSGOptiXTest 
   [ $? -ne 0 ] && echo $msg dbg error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    CSGOptiXTest 
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 

    if [ -f "$snap" ]; then 
       open $snap
       ${IPYTHON:-ipython} --pdb -i tests/CSGOptiXTest.py 
       [ $? -ne 0 ] && echo $msg ana error && exit 3 
    else
       echo $msg snap $snap does not exist 
    fi 
fi 

exit 0 


