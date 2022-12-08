#!/bin/bash -l 
usage(){ cat << EOU
U4SimtraceTest.sh
==========================

::

    N=0 ./U4SimtraceTest.sh 
    N=1 ./U4SimtraceTest.sh 


EOU
}

bin=U4SimtraceTest

export GEOM=hamaLogicalPMT
export GEOMFOLD=/tmp/$USER/opticks/GEOM/$GEOM
export BASE=$GEOMFOLD/$bin

export VERSION=${N:-0}
export FOLD=$BASE/$VERSION

export XFOLD=$GEOMFOLD/U4SimulateTest/ALL$VERSION
export XPID=${XPID:-0}


geomscript=$GEOM.sh 

if [ -f "$geomscript" ]; then  
    source $geomscript $VERSION
else
    echo $BASH_SOURCE : no gemoscript $geomscript
fi 

# python ana level presentation 
export LOC=skip


loglevels()
{
    export U4VolumeMaker=INFO
}
loglevels


log=${bin}.log
logN=${bin}_$VERSION.log

defarg="run_ana"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN    
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg"  ]; then
    [ "$arg" == "nana" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0 

