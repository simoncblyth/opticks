#!/bin/bash -l 
usage(){ cat << EOU
cxs_min.sh : minimal executable and script for shakedown
============================================================

Usage::

    MODE=2 SEL=1 ./cxs_min.sh ana 

Default SEvt being saved into:: 

    ~/.opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL/p001/

EOU
}

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

case $(uname) in
   #Linux) defarg=dbg_info ;;
   Linux) defarg=run_info ;;
   #Darwin) defarg=grab_ana ;;
   Darwin) defarg=ana ;;
esac

if [ -n "$BP" ]; then
   defarg="dbg"
fi 

arg=${1:-$defarg}

export OPTICKS_HASH=$(git -C $OPTICKS_HOME rev-parse --short HEAD)

bin=CSGOptiXSMTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

GDIR=.opticks/GEOM/$GEOM
export ${GEOM}_CFBaseFromGEOM=$HOME/$GDIR  # configure geometry to load

export BASE=$GDIR/$bin   # bin/rsync.sh special cases paths starting with . 
export EVT=${EVT:-p001}
export FOLD=$HOME/$GDIR/$bin/ALL/$EVT
export LOGDIR=$HOME/$BASE

mkdir -p $LOGDIR 
cd $LOGDIR 
LOG=$bin.log


#ipho=RainXZ_Z195_1000_f8.npy      ## ok 
#ipho=RainXZ_Z230_1000_f8.npy      ## ok
#ipho=RainXZ_Z230_10k_f8.npy       ## ok
ipho=RainXZ_Z230_100k_f8.npy
#ipho=RainXZ_Z230_X700_10k_f8.npy  ## X700 to illuminate multiple PMTs
#ipho=GridXY_X700_Z230_10k_f8.npy 
#ipho=GridXY_X1000_Z1000_40k_f8.npy

#moi=-1
#moi=sWorld:0:0
#moi=NNVT:0:0
#moi=NNVT:0:50
moi=NNVT:0:1000
#moi=PMT_20inch_veto:0:1000
#moi=sChimneyAcrylic 

# SEventConfig
export OPTICKS_INPUT_PHOTON=${OPTICKS_INPUT_PHOTON:-$ipho};
export OPTICKS_INPUT_PHOTON_FRAME=${MOI:-$moi}
export OPTICKS_EVENT_MODE=StandardFullDebug
export OPTICKS_MAX_BOUNCE=31

#export OPTICKS_MAX_PHOTON=10000
export OPTICKS_MAX_PHOTON=100000
export OPTICKS_INTEGRATION_MODE=1


# investigate double call to clear
# see ~/opticks/notes/issues/SEvt__clear_double_call.rst
#export SEvt__DEBUG_CLEAR=1

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}


logging(){ 
    export CSGOptiX=INFO
    export QEvent=INFO 
    export QSim=INFO
}
#logging



vars="GEOM LOGDIR BASE OPTICKS_HASH CVD CUDA_VISIBLE_DEVICES REALDIR FOLD"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "${arg/fold}" != "$arg" ]; then
    echo $FOLD
fi 

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then

   if [ -f "$LOG" ]; then 
       echo $BASH_SOURCE : run : delete prior LOG $LOG 
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
    ${IPYTHON:-ipython} --pdb -i $REALDIR/cxs_min.py
fi 


