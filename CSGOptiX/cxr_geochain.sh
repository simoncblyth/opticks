#!/bin/bash -l
usage(){ cat << EOU
cxr_geochain.sh 
=================

Rendering CSGFoundry geometry created from
Geant4 C++ solid specification by GeoChain 
executable. Usage::

    cd ~/opticks/GeoChain
    ./run.sh                # create geometry CF folder

    cd ~/opticks/CSGOptiX
    ./cxr_geochain.sh      # render jpg 
    EYE=0,0,1,1 ./cxr_geochain.sh  

The EYE, LOOK, UP envvars set the okc/View::home defaults 

EOU
}

msg="=== $BASH_SOURCE :"

#geometry=default
#geometry=AdditionAcrylicConstruction

geometry=pmt_solid
#geometry=body_solid
#geometry=inner_solid
#geometry=inner1_solid
#geometry=inner2_solid

#geometry=body_phys
#geometry=inner1_phys
#geometry=inner2_phys


export GEOMETRY=${GEOMETRY:-$geometry}
cfname=GeoChain/$GEOMETRY            # picks the CSGFoundry geometry to load

if [ "$GEOMETRY" == "default" ]; then  
   moi=-1
   eye=-1,0,1,1 
   tmin=0.5
   cam=0         
else
   moi=-1 
   eye=-2,0,0,1        
   tmin=1
   cam=1     # 0:perspective  1:ortho        
fi 

emm=t0    # default to no solid skips with GeoChain, which is typically single solid geometry 

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)
export MOI=${1:-$moi}
export CFNAME=${CFNAME:-$cfname}
export EMM=${EMM:-$emm}
export TMIN=${TMIN:-$tmin}
export EYE=${EYE:-$eye}
export CAM=${CAM:-$cam}    # 0:perspective 1:ortho

export NAMEPREFIX=cxr_geochain_${GEOMETRY}_   # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export RELDIR=cxr_geochain/cam_${CAM}
export TOPLINE="./cxr_geochain.sh $MOI      # EYE $EYE  $stamp  $version $GEOMETRY   " 

vars="GEOMETRY MOI CFNAME EMM TMIN EYE CAM NAMEPREFIX RELDIR TOPLINE"
echo $msg 
for var in $vars ; do printf "%-20s : %s \n" $var "${!var}" ; done  

echo $msg invoke ./cxr.sh 
./cxr.sh 

exit 0

