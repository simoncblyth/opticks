#!/bin/bash -l 

export GGeo=INFO
export CSGSolid=INFO
export CSG_GGeo_Convert=INFO

which GeoChain

if [ "$(uname)" == "Darwin" ]; then 
    lldb__ GeoChain
else
    gdb GeoChain -ex r  
fi 


