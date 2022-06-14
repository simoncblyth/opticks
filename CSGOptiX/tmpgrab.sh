#!/bin/bash -l 
usage(){ cat << EOU
tmpgrab.sh  : rsync between remote P:FOLD/EXECUTABLE and local FOLD/EXECUTABLE where FOLD is typically within /tmp/$USER/opticks 
===================================================================================================================================

Notice that the entire tmpgrab.sh approach is only appropriate for early simple geometry tests. 
Once a geometry has matured it it much better to use cachegrab.sh in order to keep 
the geometry CSGFoundry together with results obtained from the geometry.    

This script is intended to be used via controller scripts such as CSGOptiX/cxs_raindrop.sh 
which define the FOLD envvar and then sources this script in order to grab 
outputs from remote nodes::

   EXECUTABLE=CXRaindropTest source tmpgrab.sh grab

This cherry picks from the old tmp_grab.sh while following the sourced approach of cachegrab.sh 


EOU
}

tmpgrab_msg="=== $BASH_SOURCE :"
tmpgrab_arg=${1:-grab}

echo $tmpgrab_msg tmpgrab_arg $tmpgrab_arg

shift

xbase=$FOLD
xdir=$xbase/$EXECUTABLE/  ## require trailing slash to avoid rsync duplicating path element 

from=P:$xdir
to=$xdir

tmpgrab_vars="tmpgrab_arg EXECUTABLE FOLD xdir from to"
tmpgrab_dumpvars(){ for var in $tmpgrab_vars ; do printf "%-30s : %s \n" $var "${!var}" ; done ; }
tmpgrab_dumpvars


if [ "${tmpgrab_arg}" == "grab" ]; then 
    read -p "$tmpgrab_msg Enter YES to proceed with rsync between from and to " ans
    if [ "$ans" == "YES" ]; then 
        echo $tmpgrab_msg proceeding 
        mkdir -p $to
        rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
        ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
        ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `
    else
       echo $tmpgrab_msg skipping
    fi 
fi 


