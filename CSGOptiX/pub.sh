#!/bin/bash -l 
usage(){ cat << EOU

Have moved to spreading this functionality to the separate scripts 
rather than this central way 

EOU
}


msg="=== $BASH_SOURCE :"

executable=${EXECUTABLE:-CSGOptiXSimtraceTest}

if [ -n "$cfbase" ]; then 
    new_src_base=$cfbase/$EXECUTABLE
else
    opticks_key_remote_dir=$(opticks-key-remote-dir)
    new_src_base=$HOME/$opticks_key_remote_dir/CSG_GGeo/$EXECUTABLE
fi

export SRC_BASE=$new_src_base

#echo $msg SRC_BASE $SRC_BASE

pub.py $* --digestprefix 



