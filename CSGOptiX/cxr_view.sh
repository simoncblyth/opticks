#!/bin/bash -l 

#moi=sStrut      # what to look at 
moi=sWaterTube   # should be same as lLowerChimney_phys
emm=t8,
zoom=1
eye=-1,-1,-1,1

export MOI=${1:-$moi}
export EMM=${EMM:-$emm}
export ZOOM=${ZOOM:-$zoom}
export EYE=${EYE:-$eye}

export TMIN=0.4 
export CAM=0 
export QUALITY=90 


export NAMEPREFIX=cxr_view_      # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export RELDIR=cxr_view/cam_${CAM}_${EMM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)

export TOPLINE="./cxr_view.sh $MOI      # EYE $EYE EMM $EMM  $stamp  $version " 

source ./cxr.sh     

exit 0

