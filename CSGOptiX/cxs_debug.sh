#!/bin/bash -l 

usage(){ cat << EOU
cxs_debug.sh : simtrace running for sliced geometry 
=====================================================

See notes/issues/cxsim-shakedown.rst 

::

    cx
    ./cxs_debug.sh run   # remote 
    ./cxs_debug.sh grab  # local 
    ./cxs_debug.sh ana   # local python script

    ./cxs_debug.sh pvcap # local pyvista screen captures

    ./cxs_debug.sh # Darwin default is ana  

    PUB=repeated_step_point ./cxs_debug.sh pvpub

    ./cxs_debug.sh mpcap   
         # screencapture with window cropping appropriate for matplotlib 

    PUB=yellow_epsilon_looks_appropriate ./cxs_debug.sh mppub   
         # copy capture to presentation tree



::

    from opticks.ana.nbase import np_string  
    np_string(x_lpos[-3:,:3].ravel())                                                                                                                                                                                                
    Out[5]: '207.807,-50.949,113.643,206.,-44.666,105.551,185.963,124.008,-90.678'


EOU
}


moi=37684         # flat instance index obtained from cxsim p.py prd debug output of the microstep photon

#ce_offset=0,-64.59664,0    # -Y shift aligning slice plane with a cxsim photon 0 hit with microsteps 
#ce_offset=209.774,-64.59664,129.752    # center the grid on the microstep points
#ce_offset=209.774,-64.597,129.752,207.807,-50.949,113.643,206.,-44.666,105.551,185.963,124.008,-90.678

ce_offset=207.807,-50.949,113.643,206.,-44.666,105.551,185.963,124.008,-90.678

ce_scale=1   
cegs=16:0:9:500 
  
gridscale=0.1      # ordinary view covering full extent with grid
#gridscale=0.01
#gridscale=0.001
#gridscale=0.0001   # ultra closeup look 

export ZOOM=${ZOOM:-2}
export LOOK=209.774,-64.59664,129.752
export LOOKCE=${LOOKCE:-1,0.1,0.01,0.001}

export NOPVGRID=1 

# CAP_BASE depends on geometry and thus is set by "source cachegrab.sh env"
export CAP_REL=cxs_debug
export CAP_STEM=cxs_debug_moi${moi}

#export CSGFoundry_Load_writeAlt=1 

source ./cxs.sh $*

