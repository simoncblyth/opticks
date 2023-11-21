#!/bin/bash -l 
usage(){ cat << EOU
cxs_min.sh : minimal executable and script for shakedown
============================================================

Usage::

    MODE=2 SEL=1 ./cxs_min.sh ana 

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

case $(uname) in
   #Linux) defarg=dbg_info ;;
   Linux) defarg=run_info ;;
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

export EVT=${EVT:-p001}
export BASE=${TMP:-/tmp/$USER/opticks}/GEOM/$GEOM/$bin
export LOGDIR=$BASE
export FOLD=$BASE/ALL/$EVT

mkdir -p $LOGDIR 
cd $LOGDIR 
LOGFILE=$bin.log


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

export NEVT=10 


# investigate double call to clear
# see ~/opticks/notes/issues/SEvt__clear_double_call.rst
#export SEvt__DEBUG_CLEAR=1

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}


logging(){ 
    export CSGFoundry=INFO
    export CSGOptiX=INFO
    export QEvent=INFO 
    export QSim=INFO
}
[ -n "$LOG" ] && logging


vars="GEOM LOGDIR BASE OPTICKS_HASH CVD CUDA_VISIBLE_DEVICES SDIR FOLD LOG NEVT"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "${arg/fold}" != "$arg" ]; then
    echo $FOLD
fi 

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then

   if [ -f "$LOGFILE" ]; then 
       echo $BASH_SOURCE : run : delete prior LOGFILE $LOGFILE 
       rm "$LOGFILE" 
   fi 

   if [ "${arg/run}" != "$arg" ]; then
       $bin
   elif [ "${arg/dbg}" != "$arg" ]; then
       dbg__ $bin 
   fi 
   [ $? -ne 0 ] && echo $BASH_SOURCE run/dbg error && exit 1 
fi 


if [ "${arg/grab}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh $BASE
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $SDIR/cxs_min.py
fi 

