#!/bin/bash -l 

export GGeo=INFO
export CSGSolid=INFO
export CSG_GGeo_Convert=INFO

export DUMP_RIDX=0
export NTREEPROCESS_LVLIST=0

cd $OPTICKS_HOME/GeoChain
rm GeoChainTest.log 

which GeoChainTest

if [ "$(uname)" == "Darwin" ]; then 
    lldb__ GeoChainTest
else
    gdb GeoChainTest -ex r  
fi 


