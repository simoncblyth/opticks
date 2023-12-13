#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_getFrame_Test.sh
=============================

Loads a CSGFoundry geometry specified by GEOM envvar and 
accesses the frame specified by OPTICKS_INPUT_PHOTON_FRAME envvar.
The frame is used to transform some input photons and comparisons
are made between different transformation approaches. 

Examples::

    OPTICKS_INPUT_PHOTON_FRAME=PMT_20inch_veto:0:1000 ~/opticks/csg/tests/CSGFoundry_getFrame_Test.sh 


EOU
}


SDIR=$(dirname $(realpath $BASH_SOURCE))

name=CSGFoundry_getFrame_Test
script=$SDIR/$name.py 


source $DIR/../../bin/OPTICKS_INPUT_PHOTON.sh 

source ~/.opticks/GEOM/GEOM.sh  # sets GEOM envvar, edit with GEOM bash function

# export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM  # now setting this in GEOM.sh 

ipf=Hama:0:1000
export OPTICKS_INPUT_PHOTON_FRAME=${OPTICKS_INPUT_PHOTON_FRAME:-$ipf}
#export MOI=$ipf

defarg="run_ana"
arg=${1:-$defarg}

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name


if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run $name error && exit 1
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ana $script error && exit 2
fi 

exit 0 

