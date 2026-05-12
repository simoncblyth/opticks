#!/usr/bin/env bash
usage(){ cat << EOU

~/o/sysrap/tests/SEvt__createInputGenstep_configuredTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SEvt__createInputGenstep_configuredTest
which $name

defarg="info_run_ls"
arg=${1:-$defarg}



# see cxs_min.sh for source to config examples
export OPTICKS_RUNNING_MODE=SRM_TORCH
export storch_FillGenstep_type=sphere
export storch_FillGenstep_radius=100    # +ve for outwards
export storch_FillGenstep_pos=0,0,0
export storch_FillGenstep_distance=1.00 # frac_twopi control of polarization phase(tangent direction)


export OPTICKS_EVENT_MODE=Hit
export OPTICKS_HIT_MASK=EC
export OPTICKS_NUM_PHOTON=M1
export OPTICKS_NUM_EVENT=1
export OPTICKS_NUM_GENSTEP=1


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/SEvt__createInputGenstep_configuredTest

gs_name=gs.npy
export GS_NAME=${GS_NAME:-$gs_name}

case $GS_NAME in
   small.npy)  OPTICKS_NUM_PHOTON=M1 ;;
   medium.npy) OPTICKS_NUM_PHOTON=M10 ;;
   large.npy)  OPTICKS_NUM_PHOTON=M100 ;;
esac

vv="BASH_SOURCE PWD defarg arg FOLD gs_name GS_NAME OPTICKS_NUM_PHOTON"


if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE - run ERROR && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE - dbg ERROR && exit 2
fi

if [ "${arg/ls}" != "$arg" ]; then
   echo ls -alst $FOLD
   ls -alst $FOLD
fi

exit 0


