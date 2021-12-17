#!/bin/bash -l 

usage(){ cat << EOU

Hmm cxs now writing into geocache, so need to have a separate cxs_grab.sh for it 

EOU
}

default_opticks_keydir_grabbed=.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1
opticks_keydir_grabbed=${OPTICKS_KEYDIR_GRABBED:-$default_opticks_keydir_grabbed}

cgdir=$opticks_keydir_grabbed/CSG_GGeo/   ## trailing slash to avoid duplicating path element 
from=P:$cgdir
to=$HOME/$cgdir

printf "OPTICKS_KEYDIR_GRABBED %s \n " "$OPTICKS_KEYDIR_GRABBED" 
printf "opticks_keydir_grabbed %s \n " "$opticks_keydir_grabbed" 
printf "\n"
printf "cgdir  %s \n" $cgdir
printf "from   %s \n" $from 
printf "to     %s \n" $to 

mkdir -p $to

if [ "$1" != "ls" ]; then
rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
fi 

ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `





