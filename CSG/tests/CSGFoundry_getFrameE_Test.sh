#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_getFrameE_Test.sh
=============================

::
  
    ~/o/CSG/tests/CSGFoundry_getFrameE_Test.sh   
    OIPF=20000 ~/o/CSG/tests/CSGFoundry_getFrameE_Test.sh   

Loads a CSGFoundry geometry specified by GEOM envvar and 
accesses the frame specified by MOI envvar.

EOU
}


SDIR=$(dirname $(realpath $BASH_SOURCE))

name=CSGFoundry_getFrameE_Test
script=$SDIR/$name.py 

source ~/.opticks/GEOM/GEOM.sh  

moi=Hama:0:1000
#moi=sChimneyAcrylic:0:0
#moi=sChimneyAcrylic:0:-1  # gord:-1
#moi=sChimneyAcrylic:0:-2   # gord:-2 fabricate XYZ frame obtained from CE 
#moi=sChimneyAcrylic:0:-3   # gord:-3 fabricate RTP tangential frame obtained from CE
#export MOI=${MOI:-$moi}

oipf=0
export OPTICKS_INPUT_PHOTON_FRAME=${OIPF:-$oipf}


#defarg="info_run_ana"
defarg="info_run"
arg=${1:-$defarg}

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name

logging(){
   export CSGTarget=INFO 
}
[ -n "$LOG" ] && logging 


vars="BASH_SOURCE INST MOI OPTICKS_INPUT_PHOTON_FRAME"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run $name error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg $name error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ana $script error && exit 3
fi 

exit 0 

