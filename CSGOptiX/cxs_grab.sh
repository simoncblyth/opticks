#!/bin/bash -l 

cxs_grab_usage(){ cat << EOU
cxs_grab.sh
=============

Runs rsync between a remote geocache/CSG_GGeo/ directory into which cxs 
intersect "photon" arrays are persisted and local directories. 
The remote directory to grab is configurable with envvar OPTICKS_KEYDIR_GRABBED,  eg::

   .opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1

EOU
}

arg=${1:-all}

default_opticks_keydir_grabbed=.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1
opticks_keydir_grabbed=${OPTICKS_KEYDIR_GRABBED:-$default_opticks_keydir_grabbed}

cgdir=$opticks_keydir_grabbed/CSG_GGeo/   ## trailing slash to avoid duplicating path element 
from=P:$cgdir
to=$HOME/$cgdir

printf "arg %s \n" "$arg"

printf "OPTICKS_KEYDIR_GRABBED %s \n " "$OPTICKS_KEYDIR_GRABBED" 
printf "opticks_keydir_grabbed %s \n " "$opticks_keydir_grabbed" 
printf "\n"
printf "cgdir  %s \n" $cgdir
printf "from   %s \n" $from 
printf "to     %s \n" $to 

mkdir -p $to


if [ "$arg" == "png" ]; then
    rsync -zarv --progress --include="*/" --include="*.png" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.png'`

elif [ "$arg" == "all" ]; then
    rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"

    ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `
fi 




