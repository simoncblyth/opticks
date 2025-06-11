#!/bin/bash
usage(){ cat << EOU
SEvtTest.sh
============

::

   LOG=1 TEST=makeGenstepArrayFromVector ~/opticks/sysrap/tests/SEvtTest.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh

name=SEvtTest


export GEOM=SEVT_TEST
export OPTICKS_INPUT_PHOTON_FRAME=0
export CFBASE


export SEQPATH=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL4/A000/seq.npy


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

#test=CountNibbles
#test=makeGenstepArrayFromVector
test=saveExtra

export TEST=${TEST:-$test}
export FOLD=$TMP/$name/$TEST

case $TEST in
   InputPhoton) script=SEvtTestIP.py ;;
  CountNibbles) script=SEvtTestCountNibbles.py ;;
             *) script=SEvtTest.py ;;
esac


logging()
{
   type $FUNCNAME
   export SEvt=INFO
   export SEvt__LIFECYCLE=1

}
[ -n "$LOG" ] && logging

defarg=info_run_ana
arg=${1:-$defarg}


vars="BASH_SOURCE name GEOM OPTICKS_INPUT_PHOTON_FRAME SEQPATH tmp TMP test TEST FOLD script defarg arg"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python}  $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi

exit 0
