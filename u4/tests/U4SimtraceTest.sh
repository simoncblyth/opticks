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
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/$bin
export FOLD=$BASE

geomscript=$GEOM.sh 
version=${N:-0}

if [ -f "$geomscript" ]; then  
    source $geomscript $version
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
logN=${bin}_${version}.log

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
    [ "$arg" == "nana" ] && export NOGUI=1
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0 

