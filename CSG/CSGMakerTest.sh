#!/bin/bash -l 


usage(){ cat << EOU
CSGMakerTest.sh : Creates CSGFoundry directories of CSGSolid/CSGPrim/CSGNode using CSG/tests/CSGMakerTest.cc
===============================================================================================================

Used to create small test geometries, often with single solids.::

   cd ~/opticks/CSG         ## OR "c" shortcut function

   vi ~/.opticks/GEOM.txt   ## OR "geom" shortcut function 
                            ## uncomment or add GEOM name with projection suffix _XY etc..  

   ./CSGMakerTest.sh        ## reads the GEOM and runs CSGMakerTest to create the CSGFoundry  

Subsequenly visualize the geometry with::

    cd ~/opticks/CSGOptiX   ## OR "cx" shortcut 
    EYE=-1,-1,-1 ./cxr_geochain.sh       ##  reads the GEOM.txt file to pick the CSGFoundry to load

EOU
}

source ../bin/GEOM.sh trim 
bin=CSGMakerTest 

echo === $BASH_SOURCE :  GEOM $GEOM bin $bin which $(which $bin)
if [ -n "$DEBUG" ]; then 
    lldb__ $bin
else
    $bin
fi 

