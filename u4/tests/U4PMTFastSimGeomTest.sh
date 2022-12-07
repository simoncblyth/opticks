#!/bin/bash -l 
usage(){ cat << EOU
U4PMTFastSimGeomTest.sh
==========================

::

    N=0 ./U4PMTFastSimGeomTest.sh 
    N=1 ./U4PMTFastSimGeomTest.sh 


EOU
}

bin=U4PMTFastSimGeomTest

export GEOM=hamaLogicalPMT
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/$bin

## PMTFastSim/HamamatsuR12860PMTManager declProp config 
export hama_FastCoverMaterial=Cheese  
export hama_UsePMTOpticalModel=1        ## adds dynode geom 
export hama_UseNaturalGeometry=${N:-0}  ## 0:FastSim/jPOM 1:InstrumentedG4OpBoundaryProcess/CustomART

case $hama_UseNaturalGeometry in
  0) echo FastSim/jPOM ;;
  1) echo InstrumentedG4OpBoundaryProcess/CustomART ;;
esac


loglevels()
{
    export U4VolumeMaker=INFO
}
loglevels



log=${bin}.log
logN=${bin}_${hama_UseNaturalGeometry}.log

defarg="run"
arg=${1:-$defarg}

if [ "$arg" == "run" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 

    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN    
fi 

if [ "$arg" == "dbg" ]; then
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "$arg" == "ana" -o "$arg" == "nana" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nana" ] && export NOGUI=1
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0 

