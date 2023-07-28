#!/bin/bash -l 

usage(){ cat << EOU
rsync.sh 
==========

Usage::

   source ~/opticks/bin/rsync.sh /tmp/GEOM/V1J009/CSGOptiX   


EOU
}


odir=$1

if [ -n "$odir" ]; then 

    if [ -d "$odir" ]; then
        echo == $BASH_SOURCE odir $odir exists
    else
        echo == $BASH_SOURCE creating rsync destination directory $odir 
    fi 

    xdir=$odir/  ## require trailing slash to avoid rsync duplicating path element 
    from=P:$xdir

    if [ "${odir:0:1}" == "." ]; then 
        to=$HOME/$xdir
    else
        to=$xdir
    fi  

    vars="BASH_SOURCE xdir from to"
    for var in $vars ; do printf "%-30s : %s \n" $var "${!var}" ; done ; 

    mkdir -p "$to"
    rsync -zarv --progress --include="*/" \
                           --include '*.gdml' \
                           --include="*.txt" \
                           --include="*.log" \
                           --include="*.tlog" \
                           --include="*.npy" \
                           --include="*.jpg" \
                           --include="*.mp4" \
                           --include "*.json" \
                           --exclude="*" \
                           "$from" "$to"

    [ $? -ne 0 ] && echo $BASH_SOURCE rsync fail && return 1   

    tto=${to%/}  # trim the trailing slash 

    find $tto -name '*.json' -o -name '*.txt' -o -name '*.log' -o -name '*.gdml' -print0 | xargs -0 ls -1rt 
    echo == $BASH_SOURCE tto $tto jpg mp4 npy 
    find $tto -name '*.jpg' -o -name '*.mp4' -o -name '*.npy' -print0 | xargs -0 ls -1rt

else
    echo == $BASH_SOURCE no directory argument provided
fi

return 0 

