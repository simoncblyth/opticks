#!/bin/bash -l 
usage(){ cat << EOU
G4CXTest_raindrop.sh : Standalone bi-simulation with G4CXApp::Main
===================================================================

Certain special GEOM strings such as "RaindropRockAirWater" 
are recognized by U4VolumeMaker::PVS_  which is called by U4VolumeMaker::PV

For the configuratuin of the raindrop see U4VolumeMaker::RaindropRockAirWater_Configure

EOU
}

cd $(dirname $BASH_SOURCE)

bin=G4CXTest

export GEOM=RaindropRockAirWater  # GEOM is identifier for a geometry 
export U4VolumeMaker_RaindropRockAirWater_MATS=G4_Pb,G4_AIR,G4_WATER
export U4VolumeMaker_RaindropRockAirWater_HALFSIDE=100


defarg="info_run"
arg=${1:-$defarg}

vars="BASH_SOURCE GEOM" 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
       Linux) gdb -ex r --args $bin  ;;
       Darwin) lldb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2 
fi 


exit 0 

