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

case $(uname) in 
  Linux)  defarg="run"  ;;
  Darwin) defarg="ana"  ;;
esac

arg=${1:-$defarg}
bin=G4CXSimtraceTest
xbin=G4CXSimulateTest

echo $msg arg $arg bin $bin defarg $defarg

source ../bin/GEOM_.sh 

if [ -n "$CFBASE" ]; then
    BASE=$CFBASE/$bin
    X_BASE=$CFBASE/$xbin    
    UBASE=${BASE//$HOME\/}    # UBASE relative to HOME to handle rsync between different HOME
else
    BASE=/tmp/$USER/opticks/$bin/$GEOM
    X_BASE=/tmp/$USER/opticks/$xbin/$GEOM
    UBASE=$BASE
    CFBASE=$BASE
fi
# NB CFBASE is NOT exported here : it is exported for the python ana, not the C++ run 

FOLD=$BASE/ALL      # corresponds SEvt::save() with SEvt::SetReldir("ALL")
X_FOLD=$X_BASE/ALL

MOI=Hama:0:1000    
## TODO: tie MOI to OPTICKS_INPUT_PHOTON_FRAME : this means not exporting from bin/OPTICKS_INPUT_PHOTON.sh 


export MOI 



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




if [ "${arg/run}" != "$arg" ]; then 
    echo $msg run $bin
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run $bin error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
        Linux) gdb_ $bin -ex r  ;;
        Darwin) lldb__ $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg $bin error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD
    export X_FOLD
    export CFBASE
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana $bin error && exit 3 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source ../bin/rsync.sh $UBASE 
fi 

exit 0 

