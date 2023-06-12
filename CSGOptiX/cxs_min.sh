#!/bin/bash -l 
usage(){ cat << EOU
cxs_min.sh : minimal executable and script for shakedown
============================================================

   

EOU
}

DIR=$(dirname $BASH_SOURCE)

case $(uname) in
   Linux) defarg=run_info ;;
   Darwin) defarg=grab_open ;;
esac

arg=${1:-$defarg}

export OPTICKS_HASH=$(git -C $OPTICKS_HOME rev-parse --short HEAD)

bin=CSGOptiXSMTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM  # configure geometry to load

ipho=RainXZ_Z230_10k_f8.npy;
export OPTICKS_INPUT_PHOTON=${OPTICKS_INPUT_PHOTON:-$ipho};

#moi=-1
#moi=sWorld:0:0
#moi=NNVT:0:0
#moi=NNVT:0:50
#moi=NNVT:0:1000
#moi=PMT_20inch_veto:0:1000
moi=sChimneyAcrylic 

export OPTICKS_INPUT_PHOTON_FRAME=${MOI:-$moi}

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}

export PBAS=/tmp/$USER/opticks/   # note trailing slash 
export BASE=${PBAS}GEOM/$GEOM/$bin
export LOGDIR=$BASE
mkdir -p $LOGDIR 
cd $LOGDIR 

LOG=$bin.log

vars="GEOM LOGDIR BASE PBAS OPTICKS_HASH CVD CUDA_VISIBLE_DEVICES"
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
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 


