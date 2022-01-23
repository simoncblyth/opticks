#!/bin/bash -l
usage(){ cat << EOU
sdf_geochain.sh using CSGSignedDistanceFieldTest.cc and test/CSGSignedDistanceFieldTest.py
===========================================================================================

Rendering CSGFoundry geometry using SDF distance fields
and pyvista contouring techniques such as marching cubes.::

   cd ~/opticks/CSG        ## c 
   ./sdf_geochain.sh 

CAUTION: only a few primitive distance functions are implemented in csg_intersect_node.h 
This will causing missing pieces for many GEOM depending on what primitives are used.

SDF rendering of GeoChain geometries converted from G4VSolid
-------------------------------------------------------------

See GeoChain/run.sh for available GEOM names, examples::

    GEOM=XJfixtureConstruction    ./sdf_geochain.sh  
    GEOM=AltXJfixtureConstruction ./sdf_geochain.sh  

    GEOM=XJfixtureConstruction  EDGES=1  ./sdf_geochain.sh      
    GEOM=AltXJfixtureConstruction  EDGES=1  ./sdf_geochain.sh  

    ## interesting that the distance fields are clearly different (despite almost exactly the same shape)
    ## due to the different modelling from more boxes in the original fixture 

SDF rendering of Adhoc CSGMaker geometries
---------------------------------------------

See CSG/CSGMaker.cc CSGMaker::make for GEOM names::

    GEOM=UnionBoxSphere      ./sdf_geochain.sh  
    GEOM=DifferenceBoxSphere ./sdf_geochain.sh  
    GEOM=IntersectionBoxSphere ./sdf_geochain.sh  

TODO: extend to working with CSGPrim grabbed from standard geometry 
---------------------------------------------------------------------

Maybe better to do this in another script, probably use MOI for picking.

EOU
}

msg="=== $BASH_SOURCE :"

geom=AltXJfixtureConstruction
export GEOM=${GEOM:-$geom}

if [ "$(uname)" == "Linux" ]; then
    cfname=GeoChain/$GEOM    
else
    cfname=GeoChain_Darwin/$GEOM    
fi

moi=ALL

export MOI=${1:-$moi}
export CFNAME=${CFNAME:-$cfname}
cfbase=/tmp/$USER/opticks/${CFNAME}   

if [ -n "$CFNAME" -a -d "$cfbase/CSGFoundry" ]; then
   echo $msg CFNAME $CFNAME cfbase $cfbase exists : proceed with initFD_cfbase 
   export CFBASE=$cfbase
else
   echo $msg CFNAME $CFNAME cfbase $cfbase DOES NOT EXIST : proceed with adhoc CSGMaker initFD_geom
   unset CFBASE
fi

ars="GEOM MOI CFNAME CFBASE"
echo $msg 
for var in $vars ; do printf "%-20s : %s \n" $var "${!var}" ; done


bin=CSGSignedDistanceFieldTest

which $bin

if [ "$(uname)" == "Darwin" ]; then
   #lldb__ $bin
   $bin
else
   $bin
fi

[ $? -ne 0 ] && echo $msg runtime error && exit 1 

${IPYTHON:-ipython} -i tests/CSGSignedDistanceFieldTest.py 
[ $? -ne 0 ] && echo $msg ana error && exit 2

exit 0


