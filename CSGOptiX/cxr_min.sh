#!/bin/bash -l 
usage(){ cat << EOU
cxr_min.sh
===========

Shakedown cxr.sh scripts using this minimal approach. 

::

    EYE=0.2,0.2,0.2 TMIN=0.1 ./cxr_min.sh
    EYE=0.3,0.3,0.3 TMIN=0.1 ./cxr_min.sh


    EYE=10,10,10 TMIN=0.5 MOI=Hama:0:0 ./cxr_min.sh    ## invisible 
    EYE=100,100,100 TMIN=0.1 MOI=Hama:0:1000 ./cxr_min.sh 
    EYE=1000,1000,1000 TMIN=0.5 MOI=NNVT:0:0 ./cxr_min.sh  ## makes sense

HUH: suspect problem with frame targetting messing up extent units
TODO: dump the frame for debugging and save view config with renders
    

EOU
}

pkg=CSGOptiX
bin=CSGOptiXRenderTest

geom=V0J008
tmin=0.5


export CSGOptiX=INFO
export GEOM=${GEOM:-$geom}
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
export TMIN=${TMIN:-$tmin}

# as a file is written in pwd need to cd 
export TMPDIR=/tmp/$USER/opticks
export LOGDIR=$TMPDIR/$pkg/$bin
mkdir -p $LOGDIR 
cd $LOGDIR 

vars="GEOM TMIN LOGDIR"
for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 

$bin



