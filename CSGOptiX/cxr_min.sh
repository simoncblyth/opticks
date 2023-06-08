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

    EYE=0.2,0.2,0.2 TMIN=0.1 ./cxr_min.sh
    EYE=0.3,0.3,0.3 TMIN=0.1 ./cxr_min.sh


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

DIR=$(dirname $BASH_SOURCE)

case $(uname) in
   Linux) defarg=run_info ;;
   Darwin) defarg=grab_open ;;
esac

arg=${1:-$defarg}

export OPTICKS_HASH=$(git -C $OPTICKS_HOME rev-parse --short HEAD)

bin=CSGOptiXRdrTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM  # configure geometry to load


#moi=-1
#moi=sWorld:0:0
#moi=NNVT:0:0
#moi=NNVT:0:50
#moi=NNVT:0:1000
moi=PMT_20inch_veto:0:1000

#eye=1000,1000,1000
#eye=3.7878,3.7878,3.7878
#eye=-1,-1,0
#eye=-1,-1,3
eye=-1,-1,3

zoom=1
tmin=0.5

#escale=asis
escale=extent

case $moi in 
   PMT_20inch_veto:0:1000) eye=1,1,5 ; tmin=0.4  ;;
   NNVT:0:1000) eye=1,0,5 ; zoom=2 ;; 
esac

export MOI=${MOI:-$moi}
export ESCALE=${ESCALE:-$escale}
export EYE=${EYE:-$eye}
export ZOOM=${ZOOM:-$zoom}
export TMIN=${TMIN:-$tmin}

nameprefix=cxr_min_
nameprefix=${nameprefix}_eye_${EYE}_
nameprefix=${nameprefix}_zoom_${ZOOM}_
nameprefix=${nameprefix}_tmin_${TMIN}_

# moi appended within CSGOptiX::render_snap ?
export NAMEPREFIX=$nameprefix

topline="ESCALE=$ESCALE EYE=$EYE TMIN=$TMIN MOI=$MOI ZOOM=$ZOOM ~/opticks/CSGOptiX/cxr_min.sh " 
export TOPLINE=${TOPLINE:-$topline}

#export CSGFoundry=INFO 
#export CSGOptiX=INFO
# as a file is written in pwd need to cd 

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}

export PBAS=/tmp/$USER/opticks/   # note trailing slash 
export BASE=${PBAS}GEOM/$GEOM/$bin
export PPFX=${BASE}/${NAMEPREFIX}_${MOI}

export LOGDIR=$BASE
mkdir -p $LOGDIR 
cd $LOGDIR 

LOG=$bin.log

vars="GEOM TMIN LOGDIR BASE PBAS PPFX NAMEPREFIX OPTICKS_HASH TOPLINE CVD CUDA_VISIBLE_DEVICES"
#for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 


if [ "${arg/run}" != "$arg" ]; then

   if [ -f "$LOG" ]; then 
       echo $BASH_SOURCE : run : delete prior LOG $LOG 
       rm "$LOG" 
   fi 
   $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "$arg" == "grab" -o "$arg" == "open" -o "$arg" == "clean" -o "$arg" == "grab_open" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 

if [ "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    echo PBAS $PBAS
    echo PPFX $PPFX
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 


