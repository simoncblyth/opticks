#!/bin/bash -l 
usage(){ cat << EOU
cxr_overview.sh
================

::

   GDB=gdb    ./cxr_overview.sh 
   GDB=lldb__ ./cxr_overview.sh 

   EMM=0, ./cxr_overview.sh 
   EMM=1, ./cxr_overview.sh 



* cxr_overview.sh render times with JUNO 

JUNO (trunk:Dec, 2021)
   without   0.0054 

JUNO (trunk:Mar 2, 2022)

   without  0.0126
   WITH_PRD 0.0143
   without  0.0126
   without  0.0125  
   WITH_PRD 0.0143   (here and above WITH_PRD used attribs and payload values at 8 without reason)
   WITH_PRD 0.0125   (now with attribs and payload values reduced to 2)
   WITH_PRD 0.0124  

   WITH_PRD not-WITH_CONTIGUOUS 0.0123


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

emm_all=t0        # tilde zero     : (without comma so this is whole number spec)  meaning ALL
emm_noglobal=t0,  # tilde 0-th bit : (with comma meaning single bitindex spec) meaning exclude solid 0 (global) 
emm_no8=t8,       # tilde 8-th bit : exclude solid 8 
emm=$emm_all

elv=t
export EMM=${EMM:-$emm}
export ELV=${ELV:-$elv}

export NAMEPREFIX=cxr_overview_emm_${EMM}_elv_${ELV}_moi_      # MOI gets appended by the executable
export OPTICKS_RELDIR=cam_${CAM}_tmin_${TMIN}       # this can contain slashes

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)

export TOPLINE="./cxr_overview.sh    # EYE $EYE MOI $MOI ZOOM $ZOOM   $stamp  $version " 
export BOTLINE=" RELDIR $OPTICKS_RELDIR NAMEPREFIX $NAMEPREFIX SCANNER $SCANNER "

source ./cxr.sh  


