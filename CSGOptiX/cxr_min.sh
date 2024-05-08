#!/bin/bash -l 
usage(){ cat << EOU
cxr_min.sh : minimal executable and script for shakedown
============================================================

See also:

cxr_grab.sh 
   rsync pull from workstation to laptop 
   NOT USING ANYMORE : NOW DO grab HANDLING IN EACH SCRIPT TO CUSTOMIZE DIRS
     
sysrap/tests/SGLM_set_frame_test.sh 
   fast standalone SGLM::set_frame cycling using persisted sframe 


::

    EYE=0.2,0.2,0.2 TMIN=0.1 ~/o/cxr_min.sh
    EYE=0.3,0.3,0.3 TMIN=0.1 ~/o/cxr_min.sh

    ELV=t:Water_solid,Rock_solid  ./cxr_min.sh    # with colon not working 


    MOI=PMT_20inch_veto:0:1000 ~/o/cxr_min.sh



FIXED Issue : EYE inputs not being extent scaled
-----------------------------------------------------

The transition to using the transforms from sframe.h 
revealed a difference in the matrix expections, 
where the difference was extent scaling. This made
it tedious to find good viewpoints.::

    EYE=10,10,10 TMIN=0.5 MOI=Hama:0:0 ./cxr_min.sh        ## invisible 
    EYE=100,100,100 TMIN=0.1 MOI=Hama:0:1000 ./cxr_min.sh  ## mostly inviz
    EYE=1000,1000,1000 TMIN=0.5 MOI=NNVT:0:0 ./cxr_min.sh  ## makes sense

DONE: added saving of SGLM::desc for debugging view issues

    

EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

case $(uname) in
   Linux) defarg=run_info ;;
   Darwin) defarg=grab_open ;;
esac

[ -n "$BP" ] && defarg=dbg_info

arg=${1:-$defarg}


opticks_hash=$(git -C $OPTICKS_HOME rev-parse --short HEAD 2>/dev/null)
[ -z "$opticks_hash" ] && opticks_hash="FAILED_GIT_REV_PARSE" 
export OPTICKS_HASH=$opticks_hash

#bin=CSGOptiXRMTest
bin=CSGOptiXRenderInteractiveTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar, use GEOM bash function to setup/edit 
source ~/.opticks/GEOM/MOI.sh    # sets MOI envvar, use MOI bash function to setup/edit  


# TRANSITIONAL KLUDGE
export SCENE_FOLD=/tmp/SScene_test


#eye=1000,1000,1000
#eye=3.7878,3.7878,3.7878
eye=-1,-1,0
#eye=-1,-1,3
#eye=-1,-1,3


wh=2560,1440
fullscreen=1

zoom=1
tmin=0.5
cam=perspective
#cam=orthographic

#escale=asis
escale=extent

case $MOI in 
   ALL)                    eye=0,2.0,0   ; tmin=1.75 ; zoom=2.0 ;; 
   PMT_20inch_veto:0:1000) eye=1,1,5     ; tmin=0.4  ;;
   NNVT:0:1000)            eye=1,0,5     ; zoom=2 ;;
   UP_sChimneyAcrylic)     eye=-10,0,-30 ; tmin=0.1 ; zoom=0.5 ;; 
   sChimneyAcrylic*)       eye=-10,0,0   ; tmin=0.1 ; zoom=0.5 ;; 
   NEW_sChimneyAcrylic)    eye=-30,0,5   ; tmin=25  ; icam=1 ;; 
esac



export WH=${WH:-$wh}
export FULLSCREEN=${FULLSCREEN:-$fullscreen}

export ESCALE=${ESCALE:-$escale}
export EYE=${EYE:-$eye}
export ZOOM=${ZOOM:-$zoom}
export TMIN=${TMIN:-$tmin}
export CAM=${CAM:-$cam}

nameprefix=cxr_min_
nameprefix=${nameprefix}_eye_${EYE}_
nameprefix=${nameprefix}_zoom_${ZOOM}_
nameprefix=${nameprefix}_tmin_${TMIN}_

# moi appended within CSGOptiX::render_snap ?
export NAMEPREFIX=$nameprefix

topline="ESCALE=$ESCALE EYE=$EYE TMIN=$TMIN MOI=$MOI ZOOM=$ZOOM CAM=$CAM ~/opticks/CSGOptiX/cxr_min.sh " 
botline="$(date)"
export TOPLINE=${TOPLINE:-$topline}
export BOTLINE=${BOTLINE:-$botline}

logging(){
    #export CSGFoundry=INFO 
    #export CSGOptiX=INFO
    export SBT=INFO
}
[ -n "$LOG" ] && logging 


cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}


if [ -n "$TRIMESH" ]; then 

   #trimesh=2923:sWorld
   #trimesh=5:PMT_3inch_pmt_solid
   #trimesh=9:NNVTMCPPMTsMask_virtual
   #trimesh=12:HamamatsuR12860sMask_virtual
   #trimesh=6:mask_PMT_20inch_vetosMask_virtual
   #trimesh=1:sStrutBallhead
   #trimesh=1:base_steel
   trimesh=1:uni_acrylic1
   #trimesh=130:sPanel

   export OPTICKS_SOLID_TRIMESH=$trimesh
fi


TMP=${TMP:-/tmp/$USER/opticks}

export PBAS=${TMP}/    # note trailing slash 
export BASE=$TMP/GEOM/$GEOM/$bin
export LOGDIR=$BASE
mkdir -p $LOGDIR 
cd $LOGDIR 

LOG=$bin.log

vars="GEOM TMIN LOGDIR BASE PBAS NAMEPREFIX OPTICKS_HASH TOPLINE BOTLINE CVD CUDA_VISIBLE_DEVICES"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
   if [ -f "$LOG" ]; then 
       echo $BASH_SOURCE : run : delete prior LOG $LOG 
       rm "$LOG" 
   fi 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
   if [ -f "$LOG" ]; then 
       echo $BASH_SOURCE : dbg : delete prior LOG $LOG 
       rm "$LOG" 
   fi 
   gdb__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 1 
fi 

if [ "$arg" == "grab" -o "$arg" == "open" -o "$arg" == "clean" -o "$arg" == "grab_open" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 

if [ "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 


