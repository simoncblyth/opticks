#!/bin/bash -l 
notes(){ cat << EON
QEvent_Lifecycle_Test.sh
=========================

HMM: C++ 

EON
}

name=QEvent_Lifecycle_Test

logging(){
   export QEvent=INFO
}
[ -n "$LOG" ] && logging

export GEOM=TEST
export OPTICKS_INPUT_PHOTON=RainXZ_Z230_10k_f8.npy

defarg="run"
arg=${1:-$defarg}

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

exit 0 
