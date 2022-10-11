#!/bin/bash -l
usage(){ cat << EOU
x4t.sh : using X4SimtraceTest (this is an update of xxs.sh which used X4IntersectSolidTest)
=============================================================================================

Change the GEOM by editing the file opened with::

   geom_  

Aiming to follow gxt.sh such that common python plotting machinery can be used with x4t.sh outputs. 
See also cf_x4t.sh that loads and plots from multiple fold. 

::

   FOCUS=257,-39,7 ./x4t.sh ana

EOU
}

loglevels()
{
   export X4Simtrace=INFO   
   export SEvt=INFO 

   export PMTSim=3
   export VERBOSE=1
}
loglevels

arg=${1:-run_ana}

bin=X4SimtraceTest
log=$bin.log

source $(dirname $BASH_SOURCE)/../bin/COMMON.sh
unset OPTICKS_INPUT_PHOTON 
export FOLD=/tmp/$USER/opticks/GEOM/$GEOM/$bin/ALL
#export FOCUS=257,-39,7

export TOPLINE="extg4/x4t.sh GEOM $GEOM FOCUS $FOCUS" 


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

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=x4t
    export CAP_STEM=${GEOM}
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 


exit 0 
