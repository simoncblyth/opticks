#!/bin/bash -l 
usage(){ cat << EOU
GeoChain/run.sh 
================

Performs full geometry translation chain:

1. Geant4 C++ G4VSolid definition
2. NNode
3. GGeo/GPts/GParts/GMesh/GMergedMesh
4. CSGFoundry 

Usage::

   cd ~/opticks/GeoChain
   ./run.sh 

EOU
}

#name=GeoChainSolidTest
name=GeoChainVolumeTest

if [ "$name" == "GeoChainSolidTest" ]; then
    #geochaintest=AdditionAcrylicConstruction
    #geochaintest=BoxMinusTubs0
    #geochaintest=BoxMinusTubs1
    #geochaintest=PMTSim_Z
    #geochaintest=PMTSim_Zclone
    geochaintest=PMTSim_Z-400
elif [ "$name" == "GeoChainVolumeTest" ]; then
    geochaintest=PMTSimLV
else
    echo ERROR unexpected executable name $name 
    exit 1 
fi 


#export GGeo=INFO
#export CSGSolid=INFO
#export CSG_GGeo_Convert=INFO

#export NTreeProcess=INFO
#export NNodeNudger=INFO
#export NTreeBalance=INFO
#export NTreeBuilder=INFO
#export X4Solid=INFO

#export DUMP_RIDX=0
#export NTREEPROCESS_LVLIST=0
#export NNODENUDGER_LVLIST=0

export GEOCHAINTEST=${GEOCHAINTEST:-$geochaintest}

export JUNO_PMT20INCH_POLYCONE_NECK=ENABLED 
export JUNO_PMT20INCH_SIMPLIFY_CSG=ENABLED
export JUNO_PMT20INCH_NOT_USE_REAL_SURFACE=ENABLED    # when defined : dont intersect chop the PMT 

# checking that --skipsolidname is working 
export OpticksDbg=INFO  
export GInstancer=INFO


unset OPTICKS_KEY 

cd $OPTICKS_HOME/GeoChain

rm $name.log 
which $name

opts=""
opts="$opts --x4tubsnudgeskip 0"
opts="$opts --skipsolidname ${GEOCHAINTEST}_body_solid_1_9   " 


DEBUG=1

if [ -n "$DEBUG" ]; then 
    if [ "$(uname)" == "Darwin" ]; then 
        lldb__ $name $opts 
    else
        gdb -ex r --args $name $opts 
    fi
else 
    $name $opts
fi



