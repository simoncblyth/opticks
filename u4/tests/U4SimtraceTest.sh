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

## PMTFastSim/HamamatsuR12860PMTManager declProp config 
export hama_FastCoverMaterial=Cheese  
export hama_UsePMTOpticalModel=1  

version=${N:-0}
export hama_UseNaturalGeometry=$version 

case $version in
  0) echo FastSim/jPOM ;;
  1) echo InstrumentedG4OpBoundaryProcess/CustomART ;;
esac

export U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE=750  
export U4VolumeMaker_WrapAroundItem_Water_HALFSIDE=700  
export ${GEOM}_GEOMWrap=AroundCircle 
export U4VolumeMaker_MakeTransforms_AroundCircle_radius=400
export U4VolumeMaker_MakeTransforms_AroundCircle_numInRing=4
export U4VolumeMaker_MakeTransforms_AroundCircle_fracPhase=0

export LOC=skip
export CEGS=16:0:9:10


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

