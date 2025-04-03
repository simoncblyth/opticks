#!/bin/bash
usage(){ cat << EOU
cxt_min.sh : CSGOptiXTMTest Simtrace minimal executable and script for shakedown
==================================================================================

Uses CSGOptiXTMTest which just does "CSGOptiX::SimtraceMain()"
depends on GEOM to pick geometry and MOI for targetting

Workstation::

   ~/o/cxt_min.sh
   LOG=1 ~/o/cxt_min.sh

Laptop::

   ~/o/cxt_min.sh grab
   NOGRID=1 MODE=2 ~/o/cxt_min.sh ana

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
ana_script=$(realpath cxt_min.py)
SDIR=$PWD

allarg="info_fold_run_dbg_brab_grab_ana"

case $(uname) in
   Linux) defarg=run_info ;;
   Darwin) defarg=grab_ana ;;
esac

[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}

export OPTICKS_HASH=$(git -C $SDIR/.. rev-parse --short HEAD)

bin=CSGOptiXTMTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar, use GEOM bash function to setup/edit
source ~/.opticks/GEOM/MOI.sh   # sets MOI envvar, use MOI bash function to setup/edit

export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export EVT=${EVT:-A000}
export BASE=$TMP/GEOM/$GEOM
export BINBASE=$BASE/$bin
export LOGDIR=$BINBASE/$MOI
#export FOLD=$LOGDIR/$EVT
export FOLD=$TMP/GEOM/$GEOM/CSGOptiXTMTest/${MOI:-0}/$EVT
export SCRIPT=$(basename $BASH_SOURCE)

version=1
VERSION=${VERSION:-$version}

export OPTICKS_EVENT_MODE=DebugLite
export OPTICKS_INTEGRATION_MODE=1

mkdir -p $LOGDIR
cd $LOGDIR
LOGNAME=$bin.log


# pushing this too high tripped M3 max photon limit
# 16*9*2000 = 0.288 (HMM must be
export CEGS=16:0:9:2000   # XZ default
#export CEGS=16:0:9:1000   # XZ default
#export CEGS=16:0:9:100     # XZ reduce rays for faster rsync
#export CEGS=16:9:0:1000    # try XY

## base photon count without any CEHIGH for 16:0:9:2000 is (2*16+1)*(2*9+1)*2000 = 1,254,000

#export CE_OFFSET=CE    ## offsets the grid by the CE


logging(){
    type $FUNCNAME
    export CSGOptiX=INFO
    export QEvent=INFO
    #export QSim=INFO
    #export SFrameGenstep=INFO
    #export CSGTarget=INFO
    #export SEvt=INFO
    export SEvt__LIFECYCLE=INFO
}
[ -n "$LOG" ] && logging


vars="allarg defarg arg GEOM ${GEOM}_CFBaseFromGEOM MOI LOG LOGDIR BASE OPTICKS_HASH CUDA_VISIBLE_DEVICES SDIR FOLD bin script CEGS ana_script"


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
       source $SDIR/../bin/dbg__.sh
       dbg__ $bin
   fi
   [ $? -ne 0 ] && echo $BASH_SOURCE run/dbg error && exit 1
fi


if [ "${arg/brab}" != "$arg" -o "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    ## THIS OLD GRAB SYNCING TOO MUCH
    source $SDIR/../bin/BASE_grab.sh $arg
fi

if [ "${arg/grab}" != "$arg" ]; then
    source $SDIR/../bin/rsync.sh $FOLD
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $ana_script
fi


