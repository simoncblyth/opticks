#!/bin/bash 
usage(){ cat << EOU
~/o/cxd.sh : single pixel debug
=================================

Usually doing this only makes sense with
simple test geometry, or a carfully controlled
viewpoint in a full geometry. 


::
 
    SLEEP_BREAK=0 PIDXYZ=none ~/cmd.sh

    scp ~/cmd.sh A:
    scp A:cmd.sh . 



MOI=EXTENT:200
   special cased MOI for cxr_min.sh that handles
   not having normal frames with the test geometry by 
   creating a frame based on the extent valuee
   (mainly used with test geometry, as selected via GEOM
   envvar and bash function)

PIDXYZ=MIDDLE
   return debug output (only in Debug installs) for
   the pixel at the middle of the screen

PIDXYZ=-1:-1:-1
   disables debug as unsigned -1 is largest int 

PIDXYZ=0:0:0
   debug pixel at top left of render

SLEEP_BREAK=1
   sleep for one second at end of render loop then break
   use this when kernel debugging to avoid getting too much output 


EOU
}


cat $BASH_SOURCE 

moi=EXTENT:200
pidxyz=MIDDLE
sleep_break=1

export MOI=${MOI:-$moi}
export PIDXYZ=${PIDXYZ:-$pidxyz}
export SLEEP_BREAK=${SLEEP_BREAK:-$sleep_break}

~/o/cxr_min.sh

#cat $BASH_SOURCE 


