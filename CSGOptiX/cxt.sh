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

source $PWD/../bin/GEOM.sh trim   ## sets GEOM envvar based on GEOM.txt file 

if [ "$(uname)" == "Linux" ]; then
    cfname=GeoChain/$GEOM            
else
    cfname=GeoChain_Darwin/$GEOM            
fi

export CFBASE=/tmp/$USER/opticks/$cfname
export CFBASE_remote=/tmp/$USER/opticks/GeoChain/$GEOM
export FOLD=$CFBASE/CSGOptiX 
export RGMODE="render" 

if [ "$RGMODE" == "render" ]; then 

    export TMIN=0
    export EYE=-1,-1,-1
    export CAMERATYPE=1
fi 



if [ "$RGMODE" == "render" ]; then 
    snap=$FOLD/snap.jpg
    snap_remote=$CFBASE_remote/CSGOptiX/snap.jpg
fi 


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

    if [ "$RGMODE" == "render" ]; then 
        if [ -f "$snap" ]; then 
           if [ "$(uname)" == "Darwin" ]; then
               open $snap
           fi  
        else
           echo $msg snap $snap does not exist 
        fi
    fi 

    ${IPYTHON:-ipython} --pdb -i tests/CSGOptiXTest.py 
    [ $? -ne 0 ] && echo $msg ana error && exit 3 

fi 

if [ "${arg/grab}" != "$arg" ]; then

   if [ "$RGMODE" == "render" ]; then 
       mkdir -p $(dirname $snap_remote) 
       grab="scp P:$snap_remote $snap_remote"
       echo $msg $grab
       eval $grab 

       if [ "$(uname)" == "Darwin" ]; then
           open $snap_remote
       fi  
   fi 

fi 

exit 0 

