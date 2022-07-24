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


G4CXSimulateTest invokes "SEvt::save()" writing .npy to '$DefaultOutputDir/ALL' eg::

   /tmp/$USER/opticks/$GEOM/SProc::ExecutableName/ALL
   /tmp/blyth/opticks/RaindropRockAirWater/G4CXSimulateTest/ALL 

Also the CSGFoundry geometry is written to '$DefaultOutputDir/CSGFoundry' eg:: 

   /tmp/$USER/opticks/$GEOM/SProc::ExecutableName/CSGFoundry
   /tmp/blyth/opticks/RaindropRockAirWater/G4CXSimulateTest/CSGFoundry

This assumes CFBASE is not defined. When CFBASE is defined the
geometry is written to "$CFBASE/CSGFoundry" 
HMM: but when CFBASE is defined the geometry would have been loaded from there, 
hence there would be no save done ? See G4CXOpticks::setGeometry.

HMM: maybe better to distinguish loading and saving CFBASE as a form 
of control of when to save ?


EOU
}


defarg="run"
arg=${1:-$defarg}

case $arg in 
  fold) QUIET=1 ;; 
esac


bin=G4CXSimulateTest
gxsdir=$(dirname $BASH_SOURCE)
source $gxsdir/../bin/COMMON.sh 


BASE=$GEOMDIR/$bin
UBASE=${BASE//$HOME\/}    # UBASE is BASE relative to HOME to handle rsync between different HOME
FOLD=$BASE/ALL            # corresponds SEvt::save() with SEvt::SetReldir("ALL")

notes(){ cat << EON

* When BASE is not within $HOME eg its in /tmp then UBASE and BASE are the same.  

EON
}


# NB FOLD is not used by run, but it is used by ana

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE gxsdir GEOM GEOMDIR CFBASE BASE UBASE FOLD OPTICKS_INPUT_PHOTON"
    for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "$arg" == "fold" ]; then 
    echo $FOLD 
fi 


loglevels()
{
    export Dummy=INFO
    export G4CXOpticks=INFO 
    export CSGFoundry=INFO
    export U4VolumeMaker=INFO

    export SEvt=INFO
    export SOpticksKey=INFO

    #export Ctx=INFO
    #export QSim=INFO
    #export QEvent=INFO 
    #export CSGOptiX=INFO
    #export X4PhysicalVolume=INFO
}
loglevels

#export U4VolumeMaker_PVG_WriteNames=1



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
    export FOLD 
    export CFBASE=$BASE/CSGFoundry
    ${IPYTHON:-ipython} --pdb -i $gxsdir/tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana $bin error && exit 3 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source $gxsdir/../bin/rsync.sh $UBASE 
fi 

if [ "$arg" == "ab" ]; then
    cd $gxsdir
    ./gxs_ab.sh 
fi 

exit 0 

