#!/bin/bash 
usage(){ cat << EOU
QCurandStateTest.sh : testing the new chunk-centric curandState approach
=========================================================================

~/o/qudarap/tests/QCurandStateTest.sh

EOU
}

name=QCurandStateTest


gdb__ () 
{ 
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}


#defarg="info_dbg"
defarg="info_run"
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    gdb__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

exit 0 

