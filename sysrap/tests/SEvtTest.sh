#!/bin/bash -l 
usage(){ cat << EOU
SEvtTest.sh 
============

::

   ~/opticks/sysrap/tests/SEvtTest.sh 


EOU
}

name=SEvtTest

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
script=$SDIR/SEvtTestIP.py

export GEOM=SEVT_TEST
export OPTICKS_INPUT_PHOTON_FRAME=0 
export CFBASE 


export SEQPATH=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL4/A000/seq.npy 


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export FOLD=$TMP/$name


defarg=run_ana
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 




if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
fi 


