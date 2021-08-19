#!/bin/bash -l

msg="=== $BASH_SOURCE :"

usage(){ cat << EOU
::

    ./cxr_demo.sh -1    ## MOI=-1 targets CE of entire IAS i0  

    EYE=0,0,1,1 ./cxr_demo.sh -1 

The EYE, LOOK, UP envvars set the okc/View::home defaults 

Sometimes necessary to rerun creation of the Demo geometry 
for the choice of GEOMETRY, to get this to work after reboots/cleanups::

   cd ~/CSG
   ./CSGDemoTest.sh 


TODO: add "meshnames" to demo for name based MOI targetting 
TODO: pick between multiple IAS with midx -1,-2,...

EOU
}

geometry=parade
#geometry=layered_sphere
#geometry=sphere_containing_grid_of_spheres
#geometry=scaled_box3

export GEOMETRY=${GEOMETRY:-$geometry}
cfname=CSGDemoTest/$GEOMETRY            # picks the CSGFoundry geometry to load

if [ "$GEOMETRY" == "parade" ]; then    # geometry specific defaults, maybe overridden by caller scripts
   moi=0:0:4                            # what to look at 
   eye=-10,0,5,1                        # where to look from 
   tmin=0.1
   cam=0         
elif [ "$GEOMETRY" == "scaled_box3" ]; then
   moi=-1
   eye=-1,-1,0.5
   tmin=1.1 
   #tmin=1.8 
   cam=1
else
   moi=-1                               # -1 entire-IAS default as it applies to any geometry               
   eye=-1,0,1,1                         # 45 degree "tray" view 
   tmin=1
   cam=0         
fi 

emm=t0    # default to no solid skips with demo geometry 

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)
export MOI=${1:-$moi}
export CFNAME=${CFNAME:-$cfname}
export EMM=${EMM:-$emm}
export TMIN=${TMIN:-$tmin}
export EYE=${EYE:-$eye}
export CAM=${CAM:-$cam}    # 0:perspective 1:ortho

export NAMEPREFIX=cxr_demo_${GEOMETRY}_   # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export RELDIR=cxr_demo/cam_${CAM}
export TOPLINE="./cxr_demo.sh $MOI      # EYE $EYE  $stamp  $version $GEOMETRY   " 

vars="GEOMETRY MOI CFNAME EMM TMIN EYE CAM NAMEPREFIX RELDIR TOPLINE"
echo $msg 
for var in $vars ; do printf "%-20s : %s \n" $var "${!var}" ; done  

echo $msg invoke ./cxr.sh 
./cxr.sh 

exit 0

