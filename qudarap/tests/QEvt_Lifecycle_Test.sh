#!/bin/bash
notes(){ cat << EON
QEvt_Lifecycle_Test.sh
=========================

::

   ~/o/qudarap/tests/QEvt_Lifecycle_Test.sh
   LOG=1 ~/o/qudarap/tests/QEvt_Lifecycle_Test.sh
   LOG=1 BP=mkdir ~/o/qudarap/tests/QEvt_Lifecycle_Test.sh

EON
}
cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh


name=QEvt_Lifecycle_Test

logging(){
   export SEvt=INFO
   export QEvt=INFO
}
[ -n "$LOG" ] && logging

export GEOM=TEST
export OPTICKS_INPUT_PHOTON=RainXZ_Z230_10k_f8.npy
export OPTICKS_NUM_EVENT=1000
export OPTICKS_EVENT_MODE=Nothing


export FOLD=$TMP/GEOM/$GEOM/$name/ALL${VERSION:-0}

vars="BASH_SOURCE 0 PWD name GEOM FOLD"

defarg="run"
[ -n "$BP" ] && defarg=dbg

arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf " %25s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

exit 0
