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

To render the resulting geometry use eg::

   cx
   ./b7     # OptiX 7
   om       # for pre-7
  
   ./cxs.sh              # 2d python intersect render, using center-extent-gensteps
   ./cxr_geochain.sh     # 3d rendered view 


Hmm : to consolidate to single executable would need to do the below switch in PMTSim 

EOU
}
msg="=== $BASH_SOURCE :"
bin=GeoChainSolidTest   # default may be changed below

geochaintest=body_phys
export GEOCHAINTEST=${GEOCHAINTEST:-$geochaintest}

# pick the Solid or Volume binary depending on GEOCHAINTEST

case $GEOCHAINTEST in 
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

echo $msg GEOCHAINTEST $GEOCHAINTEST bin $bin


#export GGeo=INFO
#export CSGSolid=INFO
#export CSG_GGeo_Convert=INFO

#export NTreeProcess=INFO
#export NNodeNudger=INFO
#export NTreeBalance=INFO
#export NTreeBuilder=INFO

export X4Solid=INFO        # looking at G4Solid::convertEllipsoid

#export DUMP_RIDX=0
#export NTREEPROCESS_LVLIST=0
#export NNODENUDGER_LVLIST=0

export JUNO_PMT20INCH_POLYCONE_NECK=ENABLED 
export JUNO_PMT20INCH_SIMPLIFY_CSG=ENABLED
export JUNO_PMT20INCH_NOT_USE_REAL_SURFACE=ENABLED    # when defined : dont intersect chop the PMT 

# checking that --skipsolidname is working 
export OpticksDbg=INFO  
export GInstancer=INFO

unset OPTICKS_KEY 

cd $OPTICKS_HOME/GeoChain

rm $bin.log 
which $bin

opts=""
opts="$opts --x4tubsnudgeskip 0"
opts="$opts --skipsolidname ${GEOCHAINTEST}_body_solid_1_9   " 

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


