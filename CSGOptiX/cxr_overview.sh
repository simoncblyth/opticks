#!/bin/bash -l 

usage(){ cat << EOU
::

   EMM=0, ./cxr_overview.sh 
   EMM=1, ./cxr_overview.sh 



EOU
}

moi=-1
tmin=0.4
eye=-0.6,0,0,1
cam=0
zoom=1.5

export MOI=${MOI:-$moi} 
export TMIN=${TMIN:-$tmin} 
export EYE=${EYE:-$eye}
export CAM=${CAM:-$cam} 
export ZOOM=${ZOOM:-$zoom}

export QUALITY=90 
export OPTICKS_GEOM=cxr_overview

#[ "$(uname)" == "Darwin" ] && emm=1, || emm=t8,

emm=t0           # "t0" : tilde zero meaning all       "t0," : exclude bit 0 global,  "t8," exclude mm 8 
export EMM=${EMM:-$emm}

export NAMEPREFIX=cxr_overview_emm_${EMM}_moi_      # MOI gets appended by the executable
export OPTICKS_RELDIR=cam_${CAM}_tmin_${TMIN}       # this can contain slashes

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)

export TOPLINE="./cxr_overview.sh    # EYE $EYE MOI $MOI ZOOM $ZOOM   $stamp  $version " 
export BOTLINE=" RELDIR $OPTICKS_RELDIR NAMEPREFIX $NAMEPREFIX SCANNER $SCANNER "

source ./cxr.sh  


