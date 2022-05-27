#!/bin/bash -l 
usage(){ cat << EOU
cachegrab.sh 
=============

This is intended for source usage from other scripts such as cxsim.sh  

* grab.sh has become too organic : so try collecting just the essentials here 

Runs rsync grabbing into local directories files from a remote geocache/CSG_GGeo/EXECUTABLE 
directory into which executable outputs such as cxs jpg renders, json sidecars 
and intersect "photon" arrays are persisted.

The remote cache directory to grab from is configurable with envvar OPTICKS_KEY_REMOTE 
which is the OPTICKS_KEY from a remote node. The OPTICKS_KEY_REMOTE is converted
into a remote directory by bash function opticks-key-remote-dir which uses SOpticksResourceTest 
executable.

    OPTICKS_KEY_REMOTE     : $OPTICKS_KEY_REMOTE
    opticks-key-remote-dir : $(opticks-key-remote-dir)

EOU
}

msg="=== $BASH_SOURCE :"
arg=${1:-grab}
shift

if [ "$arg"  == "help" ]; then
   usage
   exit 0
fi 


executable=CSGOptiXSimTest
EXECUTABLE=${EXECUTABLE:-$executable}

opticks_key_remote_dir=$(opticks-key-remote-dir)    ## eg .opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1

xbase=$opticks_key_remote_dir/CSG_GGeo 
xdir=$xbase/$EXECUTABLE/             ## trailing slash to avoid rsync duplicating path element 

from=P:$xdir
to=$HOME/$xdir
cfbase=$HOME/$xbase
fold=$cfbase/$EXECUTABLE

printf "arg                     %s \n" "$arg"
printf "EXECUTABLE              %s \n " "$EXECUTABLE"
printf "OPTICKS_KEY_REMOTE      %s \n " "$OPTICKS_KEY_REMOTE" 
printf "opticks_key_remote_dir  %s \n " "$opticks_key_remote_dir" 
printf "\n"
printf "xdir                    %s \n" "$xdir"
printf "from                    %s \n" "$from" 
printf "to                      %s \n" "$to" 


if [ "$arg" == "env" ]; then 
    export FOLD=$fold        # used by opticks.ana.fold
    export CFBASE=$cfbase    # used by opticks.CSG.CSGFoundry 
fi 

if [ "$arg" == "grab" ]; then 
    read -p "$msg Enter YES to proceed with rsync between from and to " ans
    if [ "$ans" == "YES" ]; then 
       echo $msg proceeding 
    else
       echo $msg skipping : perhaps you should be using tmp_grab.sh 
       exit 1 
    fi 

    mkdir -p $to
    rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `
fi 


