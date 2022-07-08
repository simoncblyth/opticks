#!/bin/bash -l 
usage(){ cat << EOU
gxt.sh : G4CXSimtraceTest 
=============================================================================================================

::

    cd ~/opticks/g4cx   # gx
    ./gxt.sh 
    ./gxt.sh run
    ./gxt.sh dbg
    ./gxt.sh grab
    ./gxt.sh ana

EOU
}

msg="=== $BASH_SOURCE :"
source ../bin/GEOM_.sh 

loglevels()
{
    export Dummy=INFO
    export SEvt=INFO
    export Ctx=INFO
    #export QSim=INFO
    #export QEvent=INFO 
    export CSGOptiX=INFO
    export G4CXOpticks=INFO 
    #export X4PhysicalVolume=INFO
    #export U4VolumeMaker=INFO
}
loglevels


case $(uname) in 
  Darwin) defarg="ana"  ;;
  Linux)  defarg="run"  ;;
esac

arg=${1:-$defarg}
bin=G4CXSimtraceTest

echo $msg arg $arg bin $bin defarg $defarg

if [ "${arg/run}" != "$arg" ]; then 
    echo $msg run $bin
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
        Linux) gdb $bin -ex r  ;;
        Darwin) lldb__ $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi


export BASE=/tmp/$USER/opticks/$bin/$GEOM
export FOLD=$BASE 

if [ "${arg/ana}" != "$arg" ]; then 

    export CFBASE=$BASE
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source ../bin/rsync.sh $BASE 
fi 


exit 0 

