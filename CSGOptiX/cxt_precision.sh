#!/bin/bash
usage(){ cat << EOU
cxt_precision.sh : Simtrace Geometry Intersect Precision Test
===============================================================

~/o/CSGOptiX/cxt_precision.sh

FIG=1 TEST=input_photon_poolcover        ~/o/CSGOptiX/cxt_precision.sh pdb
FIG=1 TEST=input_photon_poolcover_refine ~/o/CSGOptiX/cxt_precision.sh pdb


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

defarg=info_run_pdb
[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}
arg2=$2

bin=CSGOptiXTMTest   ## just calls CSGOptiX::SimtraceMain()
script=cxt_precision.py
export SCRIPT=cxt_precision

source $HOME/.opticks/GEOM/GEOM.sh   # sets GEOM envvar, use GEOM bash function to setup/edit
source $HOME/.opticks/GEOM/CUR.sh 2>/dev/null  ## optionally define CUR_ bash function, for controlling directory for screenshots
source $HOME/.opticks/GEOM/EVT.sh 2>/dev/null  ## optionally define AFOLD and/or BFOLD for adding event tracks to simtrace plots




tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}


if [ "$GEOM" == "BigWaterPool" ]; then

    #test=input_photon_poolcover
    test=input_photon_poolcover_refine

    unset FIGPATH
    export TEST=${TEST:-$test}

    if [ "$TEST" == "input_photon_poolcover" -o "$TEST" == "input_photon_poolcover_refine" ]; then

        _CUR=GEOM/$GEOM/$SCRIPT/simtrace/input_photon_poolcover_and_refine

        [ -n "$FIG" ] && export FIGPATH=$HOME/Pictures/$(date +"%Y%m%d_%H%M%S").png
        export RFOLD=/data1/blyth/tmp/GEOM/BigWaterPool/CSGOptiXSMTest/ALL98_Debug_Philox_${TEST}/A000
        export OPTICKS_INPUT_PHOTON=$RFOLD/record.npy

        case $TEST in
           input_photon_poolcover)        export OPTICKS_INPUT_PHOTON_RECORD_SLICE="TO BT BT BT SA,TO BT BT SA"  ;;
           input_photon_poolcover_refine) export OPTICKS_INPUT_PHOTON_RECORD_SLICE="TO BT BT SA"                 ;;
        esac

        if [ "${TEST/refine}" != "$TEST" ]; then
            export OPTICKS_PROPAGATE_REFINE=1
            export OPTICKS_PROPAGATE_REFINE_DISTANCE=5000
        else
            unset OPTICKS_PROPAGATE_REFINE
            unset OPTICKS_PROPAGATE_REFINE_DISTANCE
        fi

        export OPTICKS_INPUT_PHOTON_RECORD_TIME="[0.1:88.8:-444]"  # -ve step for np.linspace

        ## choose first time to match TO and last time just before the earliest BT
        ## earliest first intersect r.f.record[rw,1,0,3].min() 88.851036
        export OPTICKS_EVENT_NAME=Debug_Philox_${TEST}     ## HMM cxt_precision too ?
        export AFOLD=$TMP/GEOM/$GEOM/$bin/ALL${VERSION:-0}_${OPTICKS_EVENT_NAME}/A000
    fi

fi

export OPTICKS_EVENT_MODE=DebugLite
export OPTICKS_INTEGRATION_MODE=1



logging(){
    type $FUNCNAME
    #export CSGOptiX=INFO
    #export QEvent=INFO
    #export QSim=INFO
    #export SFrameGenstep=INFO
    #export CSGTarget=INFO
    #export SEvt=INFO
    export SEvt__LIFECYCLE=INFO
    export SEvt__SIMTRACE=INFO

    export SRecord__level=3

}
[ "$LOG" == "1" ] && logging


vars="BASH_SOURCE PWD script bin defarg arg GEOM ${GEOM}_CFBaseFromGEOM CUDA_VISIBLE_DEVICES CEGS AFOLD _CUR"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/open}" != "$arg" ]; then
    # open to define current context string which controls where screenshots are copied to by ~/j/bin/pic.sh
    CUR_open ${_CUR}
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

if [ "${arg/close}" != "$arg" ]; then
    # close to invalidate the context
    CUR_close
fi

if [ "$arg" == "touch"  ]; then
    if [ -n "$arg2" ]; then
        CUR_touch "$arg2"
    else
        echo $BASH_SOURCE arg2 needs to be a datetime accepted by CUR_touch eg "cxt_precision.sh touch 21:00"
    fi
fi





