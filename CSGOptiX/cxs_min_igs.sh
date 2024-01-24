#!/bin/bash -l 
usage(){ cat << EOU
cxs_min_igs.sh : Standalone pure Opticks Optical Propagation using sequence of 1000 input genstep files 
==========================================================================================================

This was used to hunt down the cause of VRAM memory leak that incremented
memory usage by 14kb every launch.  See ~/opticks/notes/issues/okjob_GPU_memory_leak.rst 

::

   ~/o/CSGOptiX/cxs_min_igs.sh 

* skipping the launch, dont see the leak : GPU mem stays 1283 MiB
* with the launch, clear continuous growth from 1283 MiB across 1000 evt 
* skipping the gather only (not the launch) still leaking the same
* OPTICKS_MAX_BOUNCE=0 ie only generate, seems no difference to leak 
* making the launch do "nothing" with rg_dummy : curiously increases the leak by 1.7x
* scale with MAX_PHOTONS ? NO, changing MAX_PHOTONS makes no difference to size of leak 
* try default stream for the launch ? perhaps then only one stream for all launches ?

  * BINGO : USING THE DEFAULT STREAM FOR ALL optixLaunch AVOIDS THE LEAK  


Forum hunt
------------

* https://forums.developer.nvidia.com/search?q=optix%20GPU%20memory%20leak%20
* https://forums.developer.nvidia.com/search?q=optix%20memory%20
* https://forums.developer.nvidia.com/t/cudastreamcreate-calls-in-optixhello-and-optixtriangle-samples/239688


BINGO : Forum message that enabled finding the 14kb/launch VRAM leak
---------------------------------------------------------------------

* https://forums.developer.nvidia.com/t/cudastreamcreate-calls-in-optixhello-and-optixtriangle-samples/239688

CSGOptiX.cc::

    1031     if(DEBUG_SKIP_LAUNCH == false)
    1032     {
    1033         CUdeviceptr d_param = (CUdeviceptr)Params::d_param ; ;
    1034         assert( d_param && "must alloc and upload params before launch");
    1035 
    1036         /*
    1037         // this way leaking 14kb for every launch 
    1038         CUstream stream ;
    1039         CUDA_CHECK( cudaStreamCreate( &stream ) );
    1040         OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    1041         */
    1042 
    1043         // Using the default stream seems to avoid 14k VRAM leak at every launch. 
    1044         // Does that mean every launch gets to use the same single default stream ?  
    1045         CUstream stream = 0 ;
    1046         OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    1047 
    1048         CUDA_SYNC_CHECK();
    1049         // see CSG/CUDA_CHECK.h the CUDA_SYNC_CHECK does cudaDeviceSyncronize
    1050         // THIS LIKELY HAS LARGE PERFORMANCE IMPLICATIONS : BUT NOT EASY TO AVOID (MULTI-BUFFERING ETC..)  
    1051     }


* https://forums.developer.nvidia.com/t/cudastreamcreate-calls-in-optixhello-and-optixtriangle-samples/239688

kebiro
Jan 16 '23

Hi,

I have a small suggestion to make.

The above two samples issue a call to cudaStreamCreate() just before the call
to optixLaunch().

Here is where I see the problem: these two samples in particular are likely to
act as a starting point for a lot of people, so I imagine that the call to
cudaStreamCreate() can easily creep into the main loop, potentially causing a
memory leak if it’s not destroyed. At the very least this happened to me.

So I’d suggest to either add a call to cudaStreamDestroy() just before exiting
the “launch” block, or just use nullptr as the stream argument for
optixLaunch().

PS: on a side note I have another question. When running Nsight Compute via
CLI, it will print context and stream information (e.g. >kernel<, >date<,
Context 1, Stream 13). Where can I find this information in the GUI?


dhart
Moderator
Jan 16 '23

Hi @kebiro, welcome!

Thanks for the suggestion, we’ll take it under advisement. We do want to
explicitly model using CUDA streams and encourage people to know how to work
with streams as a best practice, but it’s a good point that it’s not being
explicitly cleaned up in the samples. At the very least maybe we can add a
comment.

For the Nsight Compute question, I don’t know about context and stream IDs, but
date and kernel and lots of other stats are available on the Session and
Details pages, and the kernel to inspect is available in the “Result” drop-down
near the top left, next to the “Page” drop down.

–
David.



Usage
------

(1) Start eyeballing/recording VRAM usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. Eyeballing nvidia-smi in another terminal::

    nvidia-smi -lms 500

2.  NVML memory recording with smonitor.sh::

    ~/o/sysrap/smonitor.sh       # workstation: ctrl-C after the other terminal GPU process completed


(2) Run the pure Opticks simulation using input gensteps (1000 files)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A large-ish number of events is needed so it runs for long enough (~10 seconds)
to make a recording.::

   ~/o/CSGOptiX/cxs_min_igs.sh 


(3) Stop the smonitor recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ctrl-C the smonitor process writes the smonitor.npy array and exits 


(4) Grab the recording and analyse on laptop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    ~/o/sysrap/smonitor.sh grab  # laptop

    START=10 ~/o/sysrap/smonitor.sh ana   # START specifies where to measure gradient from 

    PUB=rg_default_stream START=10 ~/o/sysrap/smonitor.sh ana
    PUB=rg_default_stream START=10 ~/o/sysrap/smonitor.sh mpcap
    PUB=rg_default_stream START=10 ~/o/sysrap/smonitor.sh mppub


Variations to pin down the leak
----------------------------------


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
|            |                                 | other than longer initialization time                  |
+------------+---------------------------------+--------------------------------------------------------+
| rg         |        0.000 GB/s               | After adopting the default CUDA Stream for optixLaunch | 
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
#export LIFECYCLE=1

#export SEvt__MINIMAL=1


SDIR=$(dirname $(realpath $BASH_SOURCE))
TEST=input_genstep $SDIR/cxs_min.sh 
