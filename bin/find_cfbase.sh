#!/bin/bash -l 

usage(){ cat << EON

Function like this is used from u4/u4s.sh 

EON
}

FOLD=/Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/ALL

upfind_cfbase(){
    : traverse directory tree upwards 
    local dir=$1
    while [ ${#dir} -gt 1 -a ! -f "$dir/CSGFoundry/solid.npy" ] ; do dir=$(dirname $dir) ; done 
    echo $dir
}

cfbase=$(upfind_cfbase $FOLD)

echo cfbase $cfbase





