#!/bin/bash
usage(){ cat << EOU
cxt_precision.sh : Simtrace Geometry Intersect Precision Test
===============================================================

~/o/CSGOptiX/cxt_precision.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))


defarg=info_run
[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}

bin=CSGOptiXTMTest   ## just calls CSGOptiX::SimtraceMain()

source $HOME/.opticks/GEOM/GEOM.sh   # sets GEOM envvar, use GEOM bash function to setup/edit
source $HOME/.opticks/GEOM/CUR.sh 2>/dev/null  ## optionally define CUR_ bash function, for controlling directory for screenshots
source $HOME/.opticks/GEOM/EVT.sh 2>/dev/null  ## optionally define AFOLD and/or BFOLD for adding event tracks to simtrace plots


if [ "$GEOM" == "BigWaterPool" ]; then

    export OPTICKS_INPUT_PHOTON=/data1/blyth/tmp/GEOM/BigWaterPool/CSGOptiXSMTest/ALL98_Debug_Philox_input_photon_poolcover/A000/record.npy
    export OPTICKS_INPUT_PHOTON_RECORD_SLICE="TO BT BT BT SA"
    export OPTICKS_INPUT_PHOTON_RECORD_TIME=80  # ns

fi



tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export OPTICKS_EVENT_MODE=DebugLite
export OPTICKS_INTEGRATION_MODE=1

mkdir -p $LOGDIR
cd $LOGDIR
LOGNAME=$bin.log


logging(){
    type $FUNCNAME
    export CSGOptiX=INFO
    export QEvent=INFO
    #export QSim=INFO
    #export SFrameGenstep=INFO
    #export CSGTarget=INFO
    #export SEvt=INFO
    export SEvt__LIFECYCLE=INFO
    export SEvt__SIMTRACE=INFO
}
[ "$LOG" == "1" ] && logging


vars="BASH_SOURCE script bin defarg arg GEOM ${GEOM}_CFBaseFromGEOM CUDA_VISIBLE_DEVICES CEGS"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run/dbg error && exit 1
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
fi

