#!/bin/bash -l 
usage(){ cat << EOU
cxt_min.sh : Simtrace minimal executable and script for shakedown
===================================================================

Workstation::

   ~/o/cxt_min.sh

Laptop::

   ~/o/cxt_min.sh grab
   NOGRID=1 MODE=2 ~/o/cxt_min.sh ana

EOU
}


SDIR=$(dirname $(realpath $BASH_SOURCE))
SNAME=$(basename $BASH_SOURCE)
SSTEM=${SNAME/.sh}
ana_script=$SDIR/$SSTEM.py 


case $(uname) in
   Linux) defarg=run_info ;;
   Darwin) defarg=grab_ana ;;
esac

[ -n "$BP" ] && defarg="info_dbg" 

arg=${1:-$defarg}

export OPTICKS_HASH=$(git -C $OPTICKS_HOME rev-parse --short HEAD)

bin=CSGOptiXTMTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

#moi=ALL
#moi=sWorld:0:0
#moi=NNVT:0:0
#moi=NNVT:0:50
#moi=NNVT:0:1000
#moi=PMT_20inch_veto:0:1000
moi=sChimneyAcrylic:0:-2    # gord:-2 XYZ frame from SCenterExtentFrame.h (now without extent scaling)

#export GRIDSCALE=100    ## WITHOUT THIS THE GRID DEFAULTS TO BEING TO SMALL 
# SUSPECT THE BELOW SETTINGS NEEDED WITH GLOBAL NON-INSTANCED VOLS 
#export CE_SCALE=1 
#export CE_OFFSET=CE


export MOI=${MOI:-$moi}  # SEventConfig


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export EVT=${EVT:-A000}
export BASE=$TMP/GEOM/$GEOM
export BINBASE=$BASE/$bin
export LOGDIR=$BINBASE/$MOI
export FOLD=$LOGDIR/$EVT

export SCRIPT=$(basename $BASH_SOURCE)

version=1
VERSION=${VERSION:-$version}   ## see below currently using VERSION TO SELECT OPTICKS_EVENT_MODE

mkdir -p $LOGDIR 
cd $LOGDIR 
LOGNAME=$bin.log

export OPTICKS_INTEGRATION_MODE=1


# pushing this too high tripped M3 max photon limit
# 16*9*2000 = 0.288 (HMM must be 
export CEGS=16:0:9:2000   # XZ default 
#export CEGS=16:0:9:1000   # XZ default 
#export CEGS=16:0:9:100     # XZ reduce rays for faster rsync
#export CEGS=16:9:0:1000    # try XY 

## base photon count without any CEHIGH for 16:0:9:2000 is (2*16+1)*(2*9+1)*2000 = 1,254,000 

#export CE_OFFSET=CE    ## offsets the grid by the CE 


cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}

logging(){ 
    #export CSGOptiX=INFO
    #export QEvent=INFO 
    #export QSim=INFO
    #export SFrameGenstep=INFO
    export CSGTarget=INFO
}
[ -n "$LOG" ] && logging


vars="GEOM LOGDIR BASE OPTICKS_HASH CVD CUDA_VISIBLE_DEVICES SDIR SNAME SSTEM FOLD script"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/fold}" != "$arg" ]; then
    echo $FOLD
fi 

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then

   if [ -f "$LOGNAME" ]; then 
       echo $BASH_SOURCE : run/dbg : delete prior LOGNAME $LOGNAME 
       rm "$LOGNAME" 
   fi 

   if [ "${arg/run}" != "$arg" ]; then
       $bin
   elif [ "${arg/dbg}" != "$arg" ]; then
       dbg__ $bin
   fi 
   [ $? -ne 0 ] && echo $BASH_SOURCE run/dbg error && exit 1 
fi 


if [ "${arg/brab}" != "$arg" -o "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    ## THIS OLD GRAB SYNCING TOO MUCH 
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source $OPTICKS_HOME/bin/rsync.sh $FOLD
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $ana_script
fi 

