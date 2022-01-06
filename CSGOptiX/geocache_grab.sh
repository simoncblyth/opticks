#!/bin/bash -l 

geocache_grab_usage(){ cat << EOU
geocache_grab.sh
==================

NB the CSG_GGeo directory is excluded as that may get large, other scripts 
handle the content of that 


EOU
}

arg=${1:-all}
shift

default_opticks_keydir_grabbed=.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1
opticks_keydir_grabbed=${OPTICKS_KEYDIR_GRABBED:-$default_opticks_keydir_grabbed}

xdir=$opticks_keydir_grabbed/   ## trailing slash to avoid duplicating path element 

from=P:$xdir
to=$HOME/$xdir


printf "arg                    %s \n" "$arg"
printf "OPTICKS_KEYDIR_GRABBED %s \n " "$OPTICKS_KEYDIR_GRABBED" 
printf "opticks_keydir_grabbed %s \n " "$opticks_keydir_grabbed" 
printf "\n"
printf "xdir                   %s \n" "$xdir"
printf "from                   %s \n" "$from" 
printf "to                     %s \n" "$to" 

mkdir -p $to


rsync  -zarv --progress --exclude="CSG_GGeo"  "$from" "$to"



