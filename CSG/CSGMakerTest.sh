#!/bin/bash -l 
usage(){ cat << EOU
CSGMakerTest.sh : Creates CSGFoundry directories of CSGSolid/CSGPrim/CSGNode using CSG/tests/CSGMakerTest.cc
===============================================================================================================

Used to create small test geometries, often with single solids.::

   cd ~/opticks/CSG         ## OR "c" shortcut function

   vi ~/.opticks/GEOM.txt   ## OR "geom" shortcut function 
                            ## uncomment or add GEOM name with projection suffix _XY etc..  

   ./CSGMakerTest.sh        ## reads the GEOM and runs CSGMakerTest to create the CSGFoundry  

EOU
}


msg="=== $BASH_SOURCE :"

geom=UnionListBoxSphere
#geom=UnionLLBoxSphere

catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && geom=$(echo ${catgeom%%_*}) 
export GEOM=${GEOM:-$geom}
bin=CSGMakerTest 

echo $msg catgeom $catgeom geom $geom GEOM $GEOM bin $bin which $(which $bin)

if [ -n "$DEBUG" ]; then 
    lldb__ $bin
else
    $bin
fi 

