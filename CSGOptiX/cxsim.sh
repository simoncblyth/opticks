#!/bin/bash -l 
usage(){ cat << EOU
cxsim.sh : CSGOptiXSimTest : standard geometry and SSim inputs 
===============================================================

Create the standard geometry::

    cg
    ./run.sh 

Run the sim using CSGOptiXSimTest executable::

    cx
    ./cxsim.sh run   # remote, "run" is default arg on Linux, "ana" on Darwin 
    ./cxsim.sh dbg   # remote, run under gdb 

    ./cxsim.sh grab  # on local, grabbing from remote
    ./cxsim.sh ana   # on local analysis using CSGOptiXSimTest.py script
  
NB the cxs_debug.sh simtrace script also loads and presents cxsim 
photons together with the simtrace geometry. 

EOU
}

msg="=== $BASH_SOURCE :"
case $(uname) in 
   Linux)  argdef=run  ;;
   Darwin) argdef=ana  ;;
esac
arg=${1:-$argdef}
bin=CSGOptiXSimTest

export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc
export OPTICKS_MAX_REC=10
export OPTICKS_MAX_SEQ=10


if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then 
    logdir=/tmp/$USER/opticks/$bin
    mkdir -p $logdir
    iwd=$PWD
    cd $logdir

    if [ "${arg/run}" != "$arg" ] ; then
        $bin
    elif [ "${arg/dbg}" != "$arg" ] ; then
        gdb $bin
    fi  

    [ $? -ne 0 ] && echo $msg RUN ERROR && exit 1 
    echo $msg logdir $logdir 
    cd $iwd
fi 


if [ "${arg/grab}" != "$arg" ]; then 
    EXECUTABLE=$bin       source cachegrab.sh grab
    EXECUTABLE=CSGFoundry source cachegrab.sh grab
fi 


if [ "${arg/ana}" != "$arg" ]; then 
    # HMM: this assumes remote running, local analysis 
    EXECUTABLE=$bin source cachegrab.sh env
    echo FOLD $FOLD 
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py  
fi 

if [ "${arg/geo}" != "$arg" ]; then 
    EXECUTABLE=$bin source cachegrab.sh env

    echo CFBASE $CFBASE

    CSGTargetTest
fi 






exit 0 
