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


opticks_key_remote_dir=$(opticks-key-remote-dir)
xdir=$opticks_key_remote_dir/   ## trailing slash to avoid duplicating path element 

from=P:$xdir
to=$HOME/$xdir


printf "arg                      %s \n" "$arg"
printf "OPTICKS_KEY_REMOTE       %s \n " "$OPTICKS_KEY_REMOTE" 
printf "opticks_key_remote_dir   %s \n " "$opticks_key_remote_dir" 
printf "\n"
printf "xdir                     %s \n" "$xdir"
printf "from                     %s \n" "$from" 
printf "to                       %s \n" "$to" 

mkdir -p $to


rsync  -zarv --progress --exclude="CSG_GGeo"  "$from" "$to"



