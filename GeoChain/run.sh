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


export GGeo=INFO
export CSGSolid=INFO
export CSG_GGeo_Convert=INFO

export DUMP_RIDX=0
export NTREEPROCESS_LVLIST=0

unset OPTICKS_KEY 

cd $OPTICKS_HOME/GeoChain
rm GeoChainTest.log 

which GeoChainTest

if [ "$(uname)" == "Darwin" ]; then 
    lldb__ GeoChainTest
else
    gdb GeoChainTest -ex r  
fi 


