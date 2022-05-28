#!/bin/bash -l 

usage(){ cat << EOU
cxs_debug.sh
=============

See notes/issues/cxsim-shakedown.rst 

::

    cx
    ./cxs_debug.sh run   # remote 
    ./cxs_debug.sh grab  # local 
    ./cxs_debug.sh ana   # local python script

    ./cxs_debug.sh pvcap # local pyvista screen captures
    ./cxs_debug.sh mpcap # local matplotlib screen captures  

    ./cxs_debug.sh # Darwin default is ana  

    PUB=repeated_step_point ./cxs_debug.sh pvpub

EOU
}


moi=37684
ce_offset=0,-64.59664,0    # -Y shift aligning slice plane with a cxsim photon 0 hit with microsteps 
ce_scale=1   
cegs=16:0:9:500   
gridscale=0.10

export ZOOM=2
export LOOK=209.774,-64.59664,129.752

# CAP_BASE depends on geometry and thus is set by "source cachegrab.sh env"
export CAP_REL=cxs_debug
export CAP_STEM=cxs_debug_moi${moi}

source ./cxs.sh $*

