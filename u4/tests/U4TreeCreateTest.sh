#!/bin/bash -l 

usage(){ cat << EOU
U4TreeCreateTest.sh 
======================

EOU
}

bin=U4TreeCreateTest 
defarg="run_ana"
arg=${1:-$defarg}

loglevels(){
   export U4VolumeMaker=INFO
   export U4Solid=INFO
}

loglevels

#geom=J006
geom=J007
export GEOM=${GEOM:-$geom}
export ${GEOM}_GDMLPath=$HOME/.opticks/GEOM/$GEOM/origin.gdml

export FOLD=/tmp/$USER/opticks/$bin


if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/load}" != "$arg" ]; then 
    $bin load
    [ $? -ne 0 ] && echo $BASH_SOURCE load error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    
    case $(uname) in
    Darwin) lldb__ $bin ;;
    Linux)  gdb__ $bin ;;
    esac

    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 


if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 4
fi 

exit 0 


