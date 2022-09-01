#!/bin/bash -l
usage(){ cat << EOU
x4t.sh : reboot of xxs.sh using X4Simtrace 
======================================================

Aiming to follow gxt.sh such that the gxt.sh python machinery 
can be used with x4t.sh outputs. 

::

   FOCUS=257,-39,7 ./x4t.sh ana

EOU
}


export GRIDSCALE=0.1  ## HMM need to align the defaults used by gxt.sh and x4t.sh 


export X4Simtrace=INFO 
export SEvt=INFO 


arg=${1:-run_ana}

#bin=X4IntersectSolidTest
bin=X4SimtraceTest
log=$bin.log

source $(dirname $BASH_SOURCE)/../bin/COMMON.sh
unset OPTICKS_INPUT_PHOTON 

export FOLD=/tmp/$USER/opticks/$GEOM/$bin/ALL


export S_GEOM=nmskSolidMask
export T_GEOM=nmskSolidMaskTail

export S_FOLD=/tmp/$USER/opticks/$S_GEOM/$bin/ALL
export T_FOLD=/tmp/$USER/opticks/$T_GEOM/$bin/ALL



if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin GEOM FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi  

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
        Darwin) lldb__ $bin ;;
        Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi  

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi 

exit 0 
