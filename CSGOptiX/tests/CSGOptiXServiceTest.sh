#!/usr/bin/env bash
usage(){ cat << EOU


~/o/CSGOptiX/tests/CSGOptiXServiceTest.sh

BP=SEvent::MakeTorchGenstep CSGOptiX/tests/CSGOptiXServiceTest.sh dbg


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

bin=CSGOptiXServiceTest
source $HOME/.opticks/GEOM/GEOM.sh


# see cxs_min.sh for source of config examples
export OPTICKS_RUNNING_MODE=SRM_TORCH
export storch_FillGenstep_type=sphere
export storch_FillGenstep_radius=100    # +ve for outwards
export storch_FillGenstep_pos=0,0,0
export storch_FillGenstep_distance=1.00 # frac_twopi control of polarization phase(tangent direction)

export OPTICKS_NUM_PHOTON=M1   ## HUH : NO EFFECT : GETTING 100 photons only
export OPTICKS_NUM_EVENT=1
export OPTICKS_NUM_GENSTEP=1


#
export OPTICKS_EVENT_MODE=Hit
export OPTICKS_HIT_MASK=EC




vars="BASH_SOURCE PWD bin GEOM defarg arg"

defarg="info_run"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi


exit 0

