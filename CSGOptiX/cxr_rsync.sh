#!/bin/bash -l 

usage(){ cat << EOU

::

   ./cxr_rsync.sh cxr_view
   ./cxr_rsync.sh cxr_solid

This will copy jpg with names starting with the NAMEPREFIX 
to the BASE directory with the source directory structure
beneath the TMPBASE dir preserved in the copy::

   TMPBASE  $TMPBASE

Using rsync like this tends to result in lots of empty dirs
in the destination. Find and delete those with::

   find $HOME/simoncblyth.bitbucket.io/env/presentation/CSGOptiXRender -type d -empty 
   find $HOME/simoncblyth.bitbucket.io/env/presentation/CSGOptiXRender -type d -empty -delete


EOU
}


BASE=$HOME/simoncblyth.bitbucket.io/env/presentation
#BASE=/tmp/env/presentation
TMPBASE=/tmp/$USER/opticks/CSGOptiX/CSGOptiXRender

NAMEPREFIX=${1:-cxr_view}
from=$TMPBASE
to=$BASE


vars="BASE TMPBASE NAMEPREFIX from to"
for var in $vars ; do printf " %20s : %s \n" $var ${!var} ; done 

find $from -name "${NAMEPREFIX}*.jpg"

read -p "Enter YES to proceed with rsync-ing those to $to : " ans 

if [ "$ans" == "YES" ]; then 

    rsync  -zarv --include="*/" --include="${NAMEPREFIX}*.jpg" --exclude="*" "$from" "$to"

    find $BASE -name "${NAMEPREFIX}*.jpg"

else
    echo SKIP
fi 



