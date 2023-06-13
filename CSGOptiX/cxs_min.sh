#!/bin/bash -l 
usage(){ cat << EOU
cxs_min.sh : minimal executable and script for shakedown
============================================================

Default SEvt being saved into:: 

    /home/blyth/.opticks/GEOM/V1J009/CSGOptiXSMTest/ALL/000/

EOU
}

REALPATH=$(find $PWD -name $(basename $BASH_SOURCE)) # absolute path
REALDIR=$(dirname $REALPATH)

case $(uname) in
   Linux) defarg=run_info ;;
   #Linux) defarg=dbg_info ;;
   #Darwin) defarg=grab_open ;;
   Darwin) defarg=run_info ;;
esac

arg=${1:-$defarg}

export OPTICKS_HASH=$(git -C $OPTICKS_HOME rev-parse --short HEAD)

bin=CSGOptiXSMTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

GDIR=.opticks/GEOM/$GEOM
export ${GEOM}_CFBaseFromGEOM=$HOME/$GDIR  # configure geometry to load

export BASE=$GDIR/$bin   # rsync special cases paths starting with . 
export EVT=${EVT:-000}
export FOLD=$HOME/$GDIR/$bin/ALL/$EVT
export LOGDIR=$HOME/$BASE

mkdir -p $LOGDIR 
cd $LOGDIR 

LOG=$bin.log



ipho=RainXZ_Z230_10k_f8.npy;
export OPTICKS_INPUT_PHOTON=${OPTICKS_INPUT_PHOTON:-$ipho};

#moi=-1
#moi=sWorld:0:0
#moi=NNVT:0:0
#moi=NNVT:0:50
moi=NNVT:0:1000
#moi=PMT_20inch_veto:0:1000
#moi=sChimneyAcrylic 

export OPTICKS_INPUT_PHOTON_FRAME=${MOI:-$moi}

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}

#export CSGOptiX=INFO
export QEvent=INFO 
export QSim=INFO


vars="GEOM LOGDIR BASE OPTICKS_HASH CVD CUDA_VISIBLE_DEVICES REALPATH REALDIR FOLD"

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


if [ "$arg" == "grab" -o "$arg" == "open" -o "$arg" == "clean" -o "$arg" == "grab_open" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 

if [ "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $REALDIR/cxs_min.py
fi 


