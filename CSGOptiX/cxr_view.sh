#!/bin/bash -l 

usage(){ cat << EOU

    EYE=0,-0.5,0.75,1 TMIN=0.5 MOI=Hama:0:1000 ./cxr_view.sh 
    
    MOI=Hama:0:1000 ./cxr_view.sh 

    MOI=NNVT:0:1000 ./cxr_view.sh 

    MOI=NNVT:0:1000 EYE=-10,-10,-10,1 ./cxr_view.sh 

    
    MOI=NNVT:0:1000 EYE=0,2,-4 ./cxr_view.sh 





   MOI=sWorld EYE=0,0.6,0.4 TMIN=0.4 ./cxr_view.sh


Nice views::

    MOI=NNVT:0:1000 EYE=0,1,-2,1 ./cxr_view.sh 

    MOI=NNVT:0:1000 EYE=0,2,-4,1 ./cxr_view.sh 

    MOI=sWaterTube EYE=0,1,-0.5 LOOK=0,0,-0.5 ./cxr_view.sh 

    MOI=sWaterTube EYE=0,1,-0.5 LOOK=0,0,-0.5 TMIN=1 ./cxr_view.sh 



    MOI=sWaterTube EYE=0,1,-1,1 LOOK=0,0,-1 ./cxr_view.sh 



EOU
}


#moi=sStrut      # what to look at 
moi=sWaterTube   # should be same as lLowerChimney_phys
emm=t0      # "t0" : tilde zero meaning all       "t0," : exclude bit 0 global,  "t8," exclude mm 8 
zoom=1
eye=-1,-1,-1,1
tmin=0.4
cam=0
quality=90

export MOI=${MOI:-$moi}
export EMM=${EMM:-$emm}
export ZOOM=${ZOOM:-$zoom}
export EYE=${EYE:-$eye}
export TMIN=${TMIN:-$tmin} 
export CAM=${CAM:-$cam} 
export QUALITY=${QUALITY:-$quality} 

nameprefix=cxr_view_${sla}_

if [ -n "$EYE" ]; then 
   nameprefix=${nameprefix}_eye_${EYE}_
fi 
if [ -n "$LOOK" ]; then 
   nameprefix=${nameprefix}_look_${LOOK}_
fi 
if [ -n "$ZOOM" ]; then 
   nameprefix=${nameprefix}_zoom_${ZOOM}_
fi 
if [ -n "$TMIN" ]; then 
   nameprefix=${nameprefix}_tmin_${TMIN}_
fi 


export NAMEPREFIX=$nameprefix               # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export OPTICKS_RELDIR=cam_${CAM}_${EMM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)

export TOPLINE="./cxr_view.sh $MOI      # EYE $EYE EMM $EMM  $stamp  $version " 

source ./cxr.sh     

exit 0

