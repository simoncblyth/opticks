#!/bin/bash -l 

usage(){ cat << EOU
cxs_debug.sh
=============

See notes/issues/cxsim-shakedown.rst 

::

    cx
    ./cxs_debug.sh run   # remote 
    ./cxs_debug.sh grab  # local 
    ./cxs_debug.sh ana   # local 


    ./cxs_debug.sh # Darwin default is ana  


EOU
}


moi=37684
ce_offset=0,-64.59664,0    # -Y shift aligning slice plane with a cxsim photon 0 hit with microsteps 
ce_scale=1   
cegs=16:0:9:500   
gridscale=0.10

export ZOOM=2
export LOOK=209.774,-64.59664,129.752

source ./cxs.sh $*

