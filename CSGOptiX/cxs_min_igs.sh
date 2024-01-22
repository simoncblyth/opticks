#!/bin/bash -l 
usage(){ cat << EOU
cxs_min_igs.sh
===============

::

   ~/o/CSGOptiX/cxs_min_igs.sh 

* skipping the launch, dont see the leak : GPU mem stays 1283 MiB
* with the launch, clear continuous growth from 1283 MiB across 1000 evt 
* skipping the gather only (not the launch) still leaking the same



* https://forums.developer.nvidia.com/search?q=optix%20GPU%20memory%20leak%20

* https://forums.developer.nvidia.com/search?q=optix%20memory%20



EOU
}


#export OPTIX_FORCE_DEPRECATED_LAUNCHER=1  ## seems no difference re leak 
#export OPTICKS_NUM_EVENT=3   # reduce from default of 1000 for shakedown
#export OPTICKS_EVENT_MODE=DebugLite 
#export QSim__simulate_DEBUG_SKIP_LAUNCH=1 
#export QSim__simulate_DEBUG_SKIP_GATHER=1 
#export CSGOptiX__launch_DEBUG_SKIP_LAUNCH=1

export OPTICKS_MAX_BOUNCE=0  ## seems no difference re leak 


SDIR=$(dirname $(realpath $BASH_SOURCE))
#TEST=input_genstep LIFECYCLE=1 $SDIR/cxs_min.sh 
TEST=input_genstep $SDIR/cxs_min.sh 
