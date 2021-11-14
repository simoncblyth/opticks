#!/bin/bash -l 
usage(){ cat << EOU
GeoChain/run.sh 
================

Performs full geometry translation chain:

1. Geant4 C++ G4VSolid OR G4VPhysicalVolume/G4LogicalVolume definition
2. X4PhysicalVolume steered conversion into NNode
3. X4PhysicalVolume steered loading into GGeo/GPts/GParts/GMesh/GMergedMesh
4. CSG_GGeo convertion of GGeo into CSGFoundry 

NB currently two executables are used, with the split decided by this
script based on the GEOM name:

GeoChainSolidTest
   for single G4VSolid

GeoChainVolumeTest
   for volumes 

HMM : consolidating to a single executable would need to implement
the below name switch in the main. But that would hardcode specific 
geometry names into code, so just stick to doing it in the script for now.
Doing this in PMTSim which is JUNO specific might be a solution.

Usage::

   gc                          # cd ~/opticks/GeoChain  
   GEOM=body_solid ./run.sh 
   GEOM=body_phys  ./run.sh 

   GEOM=body_phys ./grab.sh    # grab from remote node 


To render the resulting geometry use eg::

   cx
   ./b7     # OptiX 7
   om       # for pre-7
  
   ./cxs.sh              # 2d python intersect render, using center-extent-gensteps
   ./cxr_geochain.sh     # 3d rendered view 


EOU
}

geom=body_phys
#geom=body_solid


export GEOM=${GEOM:-$geom}
# pick the Solid or Volume binary depending on GEOM

bin=
case $GEOM in 
   AdditionAcrylicConstruction) bin=GeoChainSolidTest ;;
   BoxMinusTubs0)               bin=GeoChainSolidTest ;;
   BoxMinusTubs1)               bin=GeoChainSolidTest ;;
   UnionOfHemiEllipsoids*)      bin=GeoChainSolidTest ;;
   pmt_solid*)                  bin=GeoChainSolidTest ;;
   body_solid*)                 bin=GeoChainSolidTest ;;
   inner_solid*)                bin=GeoChainSolidTest ;;
   inner1_solid*)               bin=GeoChainSolidTest ;;
   inner2_solid*)               bin=GeoChainSolidTest ;;
   III*)                        bin=GeoChainSolidTest ;;
   1_3*)                        bin=GeoChainSolidTest ;;

   body_phys)                   bin=GeoChainVolumeTest ;;
   inner1_phys)                 bin=GeoChainVolumeTest ;; 
   inner2_phys)                 bin=GeoChainVolumeTest ;; 
   dynode_phys)                 bin=GeoChainVolumeTest ;; 
esac

msg="=== $BASH_SOURCE :"
echo $msg GEOM $GEOM bin $bin

if [ "$bin" == "" ]; then
   echo $msg ERROR do not know which executable to use for GEOM $GEOM
   exit 1 
fi


#export GGeo=INFO
#export CSGSolid=INFO
#export CSG_GGeo_Convert=INFO

#export NTreeProcess=INFO
#export NNodeNudger=INFO
#export NTreeBalance=INFO
#export NTreeBuilder=INFO

#export X4Solid=INFO        # looking at G4Solid::convertEllipsoid

#export DUMP_RIDX=0
#export NTREEPROCESS_LVLIST=0
#export NNODENUDGER_LVLIST=0

export JUNO_PMT20INCH_POLYCONE_NECK=ENABLED 
export JUNO_PMT20INCH_SIMPLIFY_CSG=ENABLED
export JUNO_PMT20INCH_NOT_USE_REAL_SURFACE=ENABLED    # when defined : dont intersect chop the PMT 
export JUNO_PMT20INCH_PLUS_DYNODE=ENABLED   # switch on dynode without new optical model

# checking that --skipsolidname is working 
export OpticksDbg=INFO  
export GInstancer=INFO

unset OPTICKS_KEY 

cd $OPTICKS_HOME/GeoChain

rm $bin.log 
which $bin

opts=""
opts="$opts --x4tubsnudgeskip 0"
#opts="$opts --skipsolidname ${GEOM}_body_solid_1_9   " 

DEBUG=1

if [ -n "$DEBUG" ]; then 
    if [ "$(uname)" == "Darwin" ]; then 
        lldb__ $bin $opts 
    else
        gdb -ex r --args $bin $opts 
    fi
else 
    $bin $opts
fi
