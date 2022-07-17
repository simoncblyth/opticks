#!/bin/bash -l 

FOLD=/Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL

upfind(){
    : traverse directory tree upwards 
    local dir=$1
    local target=${2:-CSGFoundry}
    while [ ${#dir} -gt 1 -a ! -d "$dir/$target" ] ; do dir=$(dirname $dir) ; done 
    echo $dir
}

cfbase=$(upfind $FOLD CSGFoundry)

echo cfbase $cfbase





