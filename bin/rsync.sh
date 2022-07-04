#!/bin/bash -l 

odir=$1
if [ -n "$odir" ]; then 

    xdir=$odir/  ## require trailing slash to avoid rsync duplicating path element 
    from=P:$xdir
    to=$xdir

    vars="BASH_SOURCE xdir from to"
    for var in $vars ; do printf "%-30s : %s \n" $var "${!var}" ; done ; 

    mkdir -p "$to"
    rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
    tto=${to%/}  # trim the trailing slash 

    find $tto -name '*.json' -o -name '*.txt' -print0 | xargs -0 ls -1rt 
    echo == $BASH_SOURCE tto $tto jpg mp4 npy 
    find $tto -name '*.jpg' -o -name '*.mp4' -o -name '*.npy' -print0 | xargs -0 ls -1rt

else
    echo == $BASH_SOURCE no directory argument provided
fi
