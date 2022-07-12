#!/bin/bash -l 
usage(){ cat << EOU
gxs.sh : G4CXSimulateTest : Opticks CX GPU simulation starting from Geant4 geometry auto-translated to CSG
=============================================================================================================

::

    cd ~/opticks/g4cx   # gx
    ./gxs.sh 
    ./gxs.sh info
    ./gxs.sh run
    ./gxs.sh dbg
    ./gxs.sh ana
    ./gxs.sh grab
    ./gxs.sh ab


EOU
}

defarg="run"
arg=${1:-$defarg}

case $arg in 
  fold) QUIET=1 ;; 
esac


bin=G4CXSimulateTest
source ../bin/GEOM_.sh 
source ../bin/OPTICKS_INPUT_PHOTON.sh 

if [ -n "$CFBASE" ]; then
    BASE=$CFBASE/$bin
    UBASE=${BASE//$HOME\/}    # UBASE relative to HOME to handle rsync between different HOME
else
    BASE=/tmp/$USER/opticks/$bin/$GEOM
    UBASE=$BASE
    CFBASE=$BASE
fi
# NB CFBASE is NOT exported here : it is exported for the python ana, not the C++ run 

export FOLD=$BASE/ALL      # corresponds SEvt::save() with SEvt::SetReldir("ALL")
# NB FOLD is not used by run, but it is used by ana
if [ "${arg/info}" != "$arg" ]; then 
    vars="GEOM CFBASE BASE UBASE FOLD OPTICKS_INPUT_PHOTON"
    for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "$arg" == "fold" ]; then 
    echo $FOLD 
fi 


loglevels()
{
    export Dummy=INFO
    #export U4VolumeMaker=INFO

    export SEvt=INFO
    #export Ctx=INFO
    #export QSim=INFO
    #export QEvent=INFO 
    #export CSGOptiX=INFO
    #export G4CXOpticks=INFO 
    #export X4PhysicalVolume=INFO
}
loglevels


if [ "${arg/run}" != "$arg" ]; then 
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
    export CFBASE
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana $bin error && exit 3 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source ../bin/rsync.sh $UBASE 
fi 

if [ "$arg" == "ab" ]; then
    ./gxs_ab.sh 
fi 

exit 0 

