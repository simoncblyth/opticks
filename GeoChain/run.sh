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

   PMTSIM_ZCUT=-1000 ./run.sh 

EOU
}


#geochaintest=AdditionAcrylicConstruction
#geochaintest=BoxMinusTubs0
#geochaintest=BoxMinusTubs1
#geochaintest=PMTSim_Z
#geochaintest=PMTSim_Zclone
geochaintest=PMTSim_Z-400

export GGeo=INFO
export CSGSolid=INFO
export CSG_GGeo_Convert=INFO

export NTreeProcess=INFO
export NNodeNudger=INFO
export NTreeBalance=INFO
export NTreeBuilder=INFO

export X4Solid=INFO

export DUMP_RIDX=0
export NTREEPROCESS_LVLIST=0
export NNODENUDGER_LVLIST=0

export GEOCHAINTEST=${GEOCHAINTEST:-$geochaintest}
#export PMTSIM_ZCUT=${PMTSIM_ZCUT:-$zcut}  now from name not evar 

export JUNO_PMT20INCH_POLYCONE_NECK=ENABLED 


unset OPTICKS_KEY 

cd $OPTICKS_HOME/GeoChain
rm GeoChainTest.log 

which GeoChainTest

opts="--x4tubsnudgeskip 0"


if [ "$(uname)" == "Darwin" ]; then 
    lldb__ GeoChainTest $opts 
else
    gdb -ex r --args GeoChainTest $opts 
fi 


