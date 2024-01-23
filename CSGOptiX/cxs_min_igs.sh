#!/bin/bash -l 
usage(){ cat << EOU
cxs_min_igs.sh
===============

::

   ~/o/CSGOptiX/cxs_min_igs.sh 

* skipping the launch, dont see the leak : GPU mem stays 1283 MiB
* with the launch, clear continuous growth from 1283 MiB across 1000 evt 
* skipping the gather only (not the launch) still leaking the same
* OPTICKS_MAX_BOUNCE=0 ie only generate, seems no difference to leak 

* TODO: early exit the launch, making it do "nothing" : does it still leak ?

* https://forums.developer.nvidia.com/search?q=optix%20GPU%20memory%20leak%20

* https://forums.developer.nvidia.com/search?q=optix%20memory%20


Adhoc nvidia-smi memory check in another terminal::

    nvidia-smi -lms 500

NVML memory recording::

    ~/o/sysrap/smonitor.sh       # workstation: ctrl-C after the other terminal GPU process completed

    ~/o/sysrap/smonitor.sh grab  # laptop

    PUB=with_rg_dummy START=9 ~/o/sysrap/smonitor.sh ana
    PUB=with_rg_dummy START=9 ~/o/sysrap/smonitor.sh mpcap
    PUB=with_rg_dummy START=9 ~/o/sysrap/smonitor.sh mppub


+------------+---------------------------------+--------------------------------------------------------+
| variation  | (NVML) GPU memory increase      |                                                        |
|            | across 1000 launches measured   |                                                        |
|            | at each 0.1 s                   |                                                        |
+============+=================================+========================================================+
| rg         |        0.014 GB/s               |                                                        |
+------------+---------------------------------+--------------------------------------------------------+
| rg         |        0.016, 0.015 GB/s        | after starting to add eoj cleanup (no change expected) |
+------------+---------------------------------+--------------------------------------------------------+
| rg         |        0.014 GB/s               | OPTICKS_MAX_PHOTON M1->k10 makes no difference         |
+------------+---------------------------------+--------------------------------------------------------+
| rg         |        0.015 GB/s               | OPTICKS_MAX_PHOTON M1->M10 makes no difference         |
|            |                                 | other than longer initialization time
+------------+---------------------------------+--------------------------------------------------------+
| rg_dummy   |        0.025, 0.025 GB/s        | 1.7x bigger leak with do-nothing RG ?                  |
+------------+---------------------------------+--------------------------------------------------------+


Question : How to reduce GPU memory increment at each launch ?
------------------------------------------------------------------

All the geometry+pipeline setup is done once at initialization. 
So suppose the primary memory thing happening 
at each launch is arranging stack for all the threads. 

GB and byte differences:: 

    dgb      0.125  (usedGpuMemory_GB[sel][-1]-usedGpuMemory_GB[sel][0]) 
    db  124780544.000  (usedGpuMemory[sel][-1]-usedGpuMemory[sel][0]) 

    dt        8.760  (t[sel][-1]-t[sel][0]) 

    dgb/dt       0.014  
    db/dt   14244616.140  


Divide by approx 1000 launches, gives ~14kb per launch::

    In [1]: 14244616.140/1000
    Out[1]: 14244.61614


sub-questions
~~~~~~~~~~~~~~

* scale with MAX_PHOTONS ? NO 
* scale with num photons ? 
* depend on size of geometry ? 
* try default stream for the launch ? perhaps then only one stream for all launches ?


EOU
}


#export OPTIX_FORCE_DEPRECATED_LAUNCHER=1  ## seems no difference re leak 
#export OPTICKS_NUM_EVENT=3   # reduce from default of 1000 for shakedown
#export OPTICKS_EVENT_MODE=DebugLite 
#export QSim__simulate_DEBUG_SKIP_LAUNCH=1 
#export QSim__simulate_DEBUG_SKIP_GATHER=1 
#export CSGOptiX__launch_DEBUG_SKIP_LAUNCH=1

#export OPTICKS_MAX_BOUNCE=0  ## seems no difference re leak 
#export OPTICKS_MAX_PHOTON=M10  ## change from default of 1M 


#export PIP__createRaygenPG_DUMMY=1   # replace __raygen__rg with do nothing __raygen__rg_dummy 
#export PIP=INFO 


SDIR=$(dirname $(realpath $BASH_SOURCE))
#TEST=input_genstep LIFECYCLE=1 $SDIR/cxs_min.sh 
TEST=input_genstep $SDIR/cxs_min.sh 
