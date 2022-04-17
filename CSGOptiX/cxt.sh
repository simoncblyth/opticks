#!/bin/bash -l 
usage(){ cat << EOU
cxt.sh : CSGOptiXTest : Minimal Test of CSGOptiX 
==================================================

Create GeoChain geometry::

    gc   # cd ~/opticks/GeoChain
    ./translate.sh 

Run the CSGOptiXTest::

    cx   # cd ~/opticks/CSGOptiX
    ./cxt.sh 

To grab remote snap, assuming local and remore GEOM.txt matches::

    ./cxt.sh grab 

EOU
}

msg="=== $BASH_SOURCE :"

catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && echo $msg catgeom $catgeom override of default geom $geom && geom=$(echo ${catgeom%%_*})
export GEOM=${GEOM:-$geom}


if [ "$(uname)" == "Linux" ]; then
    cfname=GeoChain/$GEOM            
else
    cfname=GeoChain_Darwin/$GEOM            
fi

export CFBASE_remote=/tmp/$USER/opticks/GeoChain/$GEOM
export CFBASE=/tmp/$USER/opticks/$cfname
export TMIN=0
export EYE=-1,-1,-1
export CAMERATYPE=1

export FOLD=$CFBASE/CSGOptiX 
snap=$FOLD/snap.jpg
snap_remote=$CFBASE_remote/CSGOptiX/snap.jpg

arg=${1:-run_ana}

if [ "${arg/dbg}" != "$arg" ]; then 
   if [ "$(uname)" == "Darwin" ]; then
       lldb__ CSGOptiXTest 
   else
       gdb CSGOptiXTest
   fi 
   [ $? -ne 0 ] && echo $msg dbg error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    CSGOptiXTest 
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 

    if [ -f "$snap" ]; then 

       if [ "$(uname)" == "Darwin" ]; then
           open $snap
       fi  

       ${IPYTHON:-ipython} --pdb -i tests/CSGOptiXTest.py 
       [ $? -ne 0 ] && echo $msg ana error && exit 3 
    else
       echo $msg snap $snap does not exist 
    fi 
fi 

if [ "${arg/grab}" != "$arg" ]; then
   mkdir -p $(dirname $snap_remote) 
   grab="scp P:$snap_remote $snap_remote"
   echo $msg $grab
   eval $grab 

   if [ "$(uname)" == "Darwin" ]; then
       open $snap_remote
   fi  
fi 

exit 0 

