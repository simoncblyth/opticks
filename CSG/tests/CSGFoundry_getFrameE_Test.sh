#!/bin/bash
usage(){ cat << EOU
CSGFoundry_getFrameE_Test.sh
=============================

::

    ~/o/CSG/tests/CSGFoundry_getFrameE_Test.sh
    OIPF=20000 ~/o/CSG/tests/CSGFoundry_getFrameE_Test.sh

Loads a CSGFoundry geometry specified by GEOM envvar and
accesses the frame specified by envvars including::

   MOI

   OPTICKS_INPUT_PHOTON_FRAME

EOU
}


name=CSGFoundry_getFrameE_Test
SDIR=$(dirname $(realpath $BASH_SOURCE))
script=$SDIR/$name.py


source ~/.opticks/GEOM/GEOM.sh







#spec=Hama:0:1000
#spec=sChimneyAcrylic:0:0
#spec=sChimneyAcrylic:0:-1  # gord:-1
#spec=sChimneyAcrylic:0:-2   # gord:-2 fabricate XYZ frame obtained from CE
#spec=sChimneyAcrylic:0:-3   # gord:-3 fabricate RTP tangential frame obtained from CE
spec=sChimneyLS:0:-2


unset MOI
unset INST
unset OPTICKS_INPUT_PHOTON_FRAME


#defarg="info_run_ana"
defarg="info_moi_ipf"
arg=${1:-$defarg}

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name

logging(){
   export CSGTarget=INFO
}
[ -n "$LOG" ] && logging


vars="BASH_SOURCE spec INST MOI OPTICKS_INPUT_PHOTON_FRAME"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run $name error && exit 1
fi

if [ "${arg/moi}" != "$arg" ]; then
    MOI=$spec $name
    [ $? -ne 0 ] && echo $BASH_SOURCE moi $name error && exit 1
fi

if [ "${arg/ipf}" != "$arg" ]; then
    OPTICKS_INPUT_PHOTON_FRAME=$spec $name
    [ $? -ne 0 ] && echo $BASH_SOURCE ipf $name error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg $name error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ana $script error && exit 3
fi

exit 0

