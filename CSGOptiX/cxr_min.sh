#!/bin/bash -l 
usage(){ cat << EOU
cxr_min.sh : minimal executable and script for shakedown
============================================================

See also:

cxr_grab.sh 
   rsync pull from workstation to laptop 
     
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

pkg=CSGOptiX
bin=CSGOptiXRdrTest

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

tmin=0.5

#moi=-1
#moi=sWorld:0:0
#moi=NNVT:0:0
moi=NNVT:0:50

#eye=1000,1000,1000
#escale=asis

#eye=3.7878,3.7878,3.7878
#eye=-1,-1,0
eye=-1,-1,3
escale=extent

export ESCALE=${ESCALE:-$escale}
export EYE=${EYE:-$eye}
export MOI=${MOI:-$moi}


export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
export TMIN=${TMIN:-$tmin}

topline="ESCALE=$ESCALE EYE=$EYE TMIN=$TMIN MOI=$MOI ~/opticks/CSGOptiX/cxr_min.sh" 
export TOPLINE=${TOPLINE:-$topline}

export CSGFoundry=INFO 
#export CSGOptiX=INFO
# as a file is written in pwd need to cd 

base=/tmp/$USER/opticks/GEOM/$GEOM/$bin
export BASE=${BASE:-$base}
export LOGDIR=$BASE
mkdir -p $LOGDIR 
cd $LOGDIR 

LOG=$bin.log

vars="GEOM TMIN LOGDIR OPTICKS_HASH TOPLINE"
for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 

if [ "${arg/run}" != "$arg" ]; then

   if [ -f "$LOG" ]; then 
       echo $BASH_SOURCE : run : delete prior LOG $LOG 
       rm "$LOG" 
   fi 
   $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
   # HMM: rename the log using the hash or some input var ? 
fi 

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 


if [ "$arg" == "grab" -o "$arg" == "open" -o "$arg" == "clean" -o "$arg" == "grab_open" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg 
fi 




