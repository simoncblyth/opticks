#!/bin/bash -l 

usage(){ cat << EOU

::

    PUB=dyn_prim_scan ./pub1.sh 

EOU
}


defpath=/Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGOptiXRenderTest/cvd1/70000/cxr_overview/cam_0_tmin_0.4/image_grid_elv_scan.jpg

img=${1:-$defpath}

relative_stem(){
   local jpg=$1
  
   local lgeocache=$HOME/.opticks/geocache/
   local geocache=${OPTICKS_GEOCACHE_PREFIX:-$HOME/.opticks}/geocache/
   local oktmp=/tmp/$USER/opticks/

   local rel 
   case $jpg in  
      ${lgeocache}*) rel=${jpg/$lgeocache/} ;;
      ${geocache}*)  rel=${jpg/$geocache/} ;;
         ${oktmp}*)  rel=${jpg/$oktmp/} ;;  
   esac 
   rel=${rel/\.jpg}

   echo $rel 
}

ext=${img##*.}
rel=$(relative_stem $img)

if [ "$PUB" == "1" ]; then 
    suf=""    ## use PUB=1 to debug the paths 
else
    suf="_${PUB}" 
fi  

s5p=/env/presentation/${rel}${suf}.${ext}
pub=$HOME/simoncblyth.bitbucket.io$s5p

echo $msg img $img 
echo $msg suf $suf 
echo $msg rel $rel 
echo $msg ext $ext 
echo $msg pub $pub
echo $msg s5p $s5p 1280px_720px 

if [ -n "$img" -a "$(uname)" == "Darwin" -a -n "$PUB" ]; then 
    mkdir -p $(dirname $pub)
    if [ -f "$pub" ]; then 
        echo $msg published path exists already : NOT COPYING : set PUB to an ext string to distinguish the name or more permanently arrange for a different path   
    elif [ "$ext" == "" ]; then 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string rather than just using PUB=1
    else
        echo $msg copying img to pub 
        cp $img $pub
        echo $msg add s5p to s5_background_image.txt
    fi
fi


