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

To render the resulting CSG geometry on GPU node use eg::

   cx
   ./b7     # OptiX 7
   om       # for pre-7
  
   ./cxs.sh              # 2D python intersect render, using center-extent-gensteps 
                         # set GEOM/geom and edit cxs.sh to configure the planar grid 
                         # of center-extent-gensteps to probe the shape in YZ XZ or XY planes
                         #

   ./cxr_geochain.sh     # 3d rendered view 


EOU
}

#geom=body_phys
#geom=body_phys_pcnk_pdyn
#geom=body_solid

#geom=hmsk_solidMask
#geom=hmsk_solidMaskTail
#geom=nmsk_solidMask
#geom=nmsk_solidMaskTail

geom=XJfixtureConstruction

#geom=SphereWithPhiSegment
#geom=PolyconeWithMultipleRmin
#geom=Orb

export GEOM=${GEOM:-$geom}
# pick the Solid or Volume binary depending on GEOM

bin=
case $GEOM in 
   SphereWithPhiSegment*)       bin=GeoChainSolidTest ;; 
   AdditionAcrylicConstruction) bin=GeoChainSolidTest ;;
   BoxMinusTubs0)               bin=GeoChainSolidTest ;;
   BoxMinusTubs1)               bin=GeoChainSolidTest ;;
   UnionOfHemiEllipsoids*)      bin=GeoChainSolidTest ;;
   PolyconeWithMultipleRmin*)   bin=GeoChainSolidTest ;; 
   pmt_solid*)                  bin=GeoChainSolidTest ;;
   body_solid*)                 bin=GeoChainSolidTest ;;
   inner_solid*)                bin=GeoChainSolidTest ;;
   inner1_solid*)               bin=GeoChainSolidTest ;;
   inner2_solid*)               bin=GeoChainSolidTest ;;
   III*)                        bin=GeoChainSolidTest ;;
   1_3*)                        bin=GeoChainSolidTest ;;

   body_phys*)                  bin=GeoChainVolumeTest ;;
   inner1_phys)                 bin=GeoChainVolumeTest ;; 
   inner2_phys)                 bin=GeoChainVolumeTest ;; 
   dynode_phys)                 bin=GeoChainVolumeTest ;; 

   *)                           bin=GeoChainSolidTest  ;;    # default : assume solid
esac


if [ "${GEOM/SphereWithPhiSegment}" != "$GEOM" ] ; then

  
   export X4Solid_convertSphere_enable_phi_segment=1 

   #return_segment=1
   #return_union=2
   #return_difference=3
   #return_intersect=4
   #return_intersect_old=14  

   #export X4Solid_intersectWithPhiSegment_debug_mode=$return_intersect_old
elif [ "${GEOM/PolyconeWithMultipleRmin}" != "$GEOM" ] ; then

   #return_inner=1
   #return_outer=2
   #export X4Solid_convertPolycone_debug_mode=$return_outer
   echo -n 

elif [ "${GEOM/XJfixtureConstruction}" != "$GEOM" ]; then

    source ../extg4/XJfixtureConstruction.sh  || exit 1 
fi 



env | grep X4Solid



msg="=== $BASH_SOURCE :"
echo $msg GEOM $GEOM bin $bin

if [ "$bin" == "" ]; then
   echo $msg ERROR do not know which executable to use for GEOM $GEOM
   exit 1 
fi

############### logging control ###################

#export GGeo=INFO
#export CSGSolid=INFO
#export CSG_GGeo_Convert=INFO

#export NTreeProcess=INFO
#export NNodeNudger=INFO
#export NTreeBalance=INFO
#export NTreeBuilder=INFO

export X4Solid=INFO        # looking at G4Solid::convertEllipsoid

# checking that --skipsolidname is working 
export OpticksDbg=INFO  
export GInstancer=INFO

#export DUMP_RIDX=0
#export NTREEPROCESS_LVLIST=0
#export NNODENUDGER_LVLIST=0


################# mechanics ###############

unset OPTICKS_KEY      # TODO: do this inside executables, as kinda important 

#####################################################

cd $OPTICKS_HOME/GeoChain

if [ -f "$bin.log" ]; then 
    rm $bin.log 
fi 

which $bin

opts=""
#opts="$opts --x4tubsnudgeskip 0"
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

exit 0

