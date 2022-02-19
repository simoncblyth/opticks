#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

geom=UnionListBoxSphere
#geom=UnionLLBoxSphere

catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && geom=$(echo ${catgeom%%_*}) 
export GEOM=${GEOM:-$geom}

echo $msg catgeom $catgeom geom $geom GEOM $GEOM


if [ -n "$DBG" ]; then 
    lldb__ CSGMakerTest 
else
    CSGMakerTest 
fi 

