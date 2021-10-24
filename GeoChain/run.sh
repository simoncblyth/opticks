#!/bin/bash -l 

export GGeo=INFO
export CSGSolid=INFO
export CSG_GGeo_Convert=INFO


cd $OPTICKS_HOME/GeoChain
rm GeoChain.log 

which GeoChain

if [ "$(uname)" == "Darwin" ]; then 
    lldb__ GeoChain
else
    gdb GeoChain -ex r  
fi 


