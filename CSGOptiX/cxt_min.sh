#!/bin/bash -l 
usage(){ cat << EOU
cxt_min.sh : Simtrace minimal executable and script for shakedown
===================================================================

EOU
}

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
REALNAME=$(basename $BASH_SOURCE)
REALSTEM=${REALNAME/.sh}

case $(uname) in
   Linux) defarg=run_info ;;
   #Linux) defarg=dbg_info ;;
   Darwin) defarg=grab_ana ;;
esac

arg=${1:-$defarg}

export OPTICKS_HASH=$(git -C $OPTICKS_HOME rev-parse --short HEAD)

bin=CSGOptiXTMTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

GDIR=.opticks/GEOM/$GEOM
export ${GEOM}_CFBaseFromGEOM=$HOME/$GDIR  # configure geometry to load

export BASE=$GDIR/$bin   # rsync special cases paths starting with . 
export EVT=${EVT:-p001}
export LOGDIR=$HOME/$BASE

mkdir -p $LOGDIR 
cd $LOGDIR 
LOG=$bin.log

moi=ALL
#moi=sWorld:0:0
#moi=NNVT:0:0
#moi=NNVT:0:50
#moi=NNVT:0:1000
#moi=PMT_20inch_veto:0:1000
#moi=sChimneyAcrylic 

# SEventConfig
export MOI=${MOI:-$moi}
#export OPTICKS_EVENT_MODE=DebugLite
#export OPTICKS_MAX_BOUNCE=31

export OPTICKS_INTEGRATION_MODE=1


## HMM: simtrace replaced ALL with the MOI value  MAYBE: STANDARDIZE ON ALL ?
export FOLD=$HOME/$GDIR/$bin/$MOI/$EVT


#export CEGS=16:0:9:1000   # XZ default 
export CEGS=16:0:9:100     # XZ reduce rays for faster rsync
#export CEGS=16:9:0:1000    # try XY 

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}

logging(){ 
    export CSGOptiX=INFO
    export QEvent=INFO 
    export QSim=INFO
}
#logging


vars="GEOM LOGDIR BASE OPTICKS_HASH CVD CUDA_VISIBLE_DEVICES REALDIR REALNAME REALSTEM FOLD"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/fold}" != "$arg" ]; then
    echo $FOLD
fi 

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then

   if [ -f "$LOG" ]; then 
       echo $BASH_SOURCE : run/dbg : delete prior LOG $LOG 
       rm "$LOG" 
   fi 

   if [ "${arg/run}" != "$arg" ]; then
       $bin
   elif [ "${arg/dbg}" != "$arg" ]; then
       case $(uname) in
          Linux) gdb__ $bin ;;
          Darwin) lldb__ $bin ;;  
       esac
   fi 

   [ $? -ne 0 ] && echo $BASH_SOURCE run/dbg error && exit 1 
fi 

if [ "${arg/grab}" != "$arg" -o "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $REALDIR/$REALSTEM.py
fi 

