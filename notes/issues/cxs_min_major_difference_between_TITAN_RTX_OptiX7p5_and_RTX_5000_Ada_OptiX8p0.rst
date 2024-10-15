cxs_min_major_difference_between_TITAN_RTX_OptiX7p5_and_RTX_5000_Ada_OptiX8p0
==================================================================================


Issue : major simulation differences between P and A (different CUDA and OptiX and GPU)
-----------------------------------------------------------------------------------------

Comparing:

* P : NVIDIA TITAN RTX, OptiX 7.5
* A : NVIDIA RTX 5000 Ada Generation, OptiX 8.0


Current hypothesis of whats gone wrong
----------------------------------------

HMM. Likely the issue in incompatibility between curandState between curand versions. 


SRM_TORCH
-----------

Running cxs_min.sh in SRM_TORCH mode on A is unrealistically fast, 
and gives no hits. 


SRM_INPUT_PHOTON
-----------------

Comparing input photon targetting NNVT:0:1000 shows hits in both P and A BUT:

* A : all hits are onto the target PMT with no others
* P : lots of other PMTs hit from reflections off the target PMT 

It looks like no reflection off the target PMT are happening for A ?


HMM: is the A build without Custom4 ? That could explain it. 

* not so, simple the Custom4 external is configured



Do some PIDX comparison between P and A
-----------------------------------------

* note the same photon start position, but are getting different randoms ? 

::

    A[blyth@localhost CSGOptiX]$ PIDX=0 ./cxs_min.sh


    //qsim.propagate.head idx 0 : bnc 0 cosTheta -0.80563819 
    //qsim.propagate.head idx 0 : mom = np.array([-0.16308457,0.53761774,0.82726693]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 0 : pos = np.array([-3191.91016,10522.31836,15746.38477]) ; lpos = 19205.62695312 
    //qsim.propagate.head idx 0 : nrm = np.array([(-0.01087651,0.03585108,-0.99929798]) ; lnrm = 1.00000000  
    //qsim.propagate_to_boundary.head idx 0 : u_absorption 0.00033755 logf(u_absorption) -7.99380875 absorption_length 41631.9062 absorption_distance 332797.500000 
    //qsim.propagate_to_boundary.head idx 0 : post = np.array([-3191.91016,10522.31836,15746.38477,   0.10000]) 


    P[blyth@localhost CSGOptiX]$ PIDX=0 ./cxs_min.sh 

    //qsim.propagate.head idx 0 : bnc 0 cosTheta -0.80563819 
    //qsim.propagate.head idx 0 : mom = np.array([-0.16308457,0.53761774,0.82726693]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 0 : pos = np.array([-3191.91016,10522.31836,15746.38477]) ; lpos = 19205.62695312 
    //qsim.propagate.head idx 0 : nrm = np.array([(-0.01087651,0.03585108,-0.99929798]) ; lnrm = 1.00000000  
    //qsim.propagate_to_boundary.head idx 0 : u_absorption 0.15698862 logf(u_absorption) -1.85158193 absorption_length 41631.9062 absorption_distance 77084.882812 
    //qsim.propagate_to_boundary.head idx 0 : post = np.array([-3191.91016,10522.31836,15746.38477,   0.10000]) 
    //qsim.propagate_to_boundary.head idx 0 : distance_to_boundary   122.6315 absorption_distance 77084.8828 scattering_distance 142337.5469 
    //qsim.propagate_to_boundary.head idx 0 : u_scattering     0.5170 u_absorption     0.1570 
     



A ems 4
---------

::

    //qsim.propagate.body.WITH_CUSTOM4 idx 0  BOUNDARY ems 4 lposcost   0.118 
    //qsim::propagate_at_surface_CustomART idx       0 : mom = np.array([-0.11694922,0.38552967,0.91525394]) ; lmom = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx       0 : pol = np.array([-0.95693922,-0.29028833,0.00000160]) ; lpol = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx       0 : nrm = np.array([-0.19764146,0.65153337,-0.73242205]) ; lnrm = 0.99999994 
    //qsim::propagate_at_surface_CustomART idx       0 : cross_mom_nrm = np.array([-0.87868893,-0.26654831,0.00000033]) ; lcross_mom_nrm = 0.91822779  
    //qsim::propagate_at_surface_CustomART idx       0 : dot_pol_cross_mom_nrm = 0.91822773 
    //qsim::propagate_at_surface_CustomART idx       0 : minus_cos_theta = -0.39605269 
    //qsim::propagate_at_surface_CustomART idx 0 lpmtid 1425 wl 440.000 mct  -0.396 dpcmn   0.918 pre-ARTE 
    //qsim::propagate_at_surface_CustomART idx 0 lpmtid 1425 wl 440.000 mct  -0.396 dpcmn   0.918 ARTE (   0.818   1.000   0.000   0.541 ) 
    //qsim.propagate_at_surface_CustomART idx 0 lpmtid 1425 ARTE (   0.818   1.000   0.000   0.541 ) u_theAbsorption    0.002 action 1 
    //qsim.propagate.tail idx 0 bounce 4 command 1 flag 64 ctx.s.optical.y(ems) 4 





Switch to DebugLite with VERSION=98 to get record, seq for history comparison
--------------------------------------------------------------------------------

P,A::

    VERSION=98 ~/o/cxs_min.sh 


Laptop::

    vip # set REMOTE to P then A and do the below

    VERSION=98 ~/o/cxs_min.sh gevt   ## rsync the event to laptop
    VERSION=98 ~/o/cxs_min.sh ana    ## pyvista analysis


* A : all 100k "TO BT BT BT BT SD" 
* P : 33k are that : but also 28k ending SA and variety of other histories

With P analysis select the most common history photons::

   w = a.q_startswith("TO BT BT BT BT SD")


Use PIDX dumping to look at the tail situation in both P and A::

    VERSION=98 PIDX=1 ~/o/cxs_min.sh run 
    VERSION=98 PIDX=4 ~/o/cxs_min.sh run 
    VERSION=98 PIDX=6 ~/o/cxs_min.sh run 
    VERSION=98 PIDX=13 ~/o/cxs_min.sh run 

Note that theEfficiency matches between A and P BUT A is always giving u_theEfficiency of 0.003::

    //qsim::propagate_at_surface_CustomART idx 13 lpmtid 1425 wl 440.000 mct  -0.397 dpcmn   0.918 ARTE (   0.819   1.000   0.000   0.541 ) 
    //qsim.propagate_at_surface_CustomART idx 13 lpmtid 1425 ARTE (   0.819   1.000   0.000   0.541 ) u_theAbsorption    0.002 action 1 
    //qsim.propagate_at_surface_CustomART.BREAK.SD/SA idx 13 lpmtid 1425 ARTE (   0.819   1.000   0.000   0.541 ) u_theEfficiency    0.003 theEfficiency   0.541 flag 64 
    //qsim.propagate.tail idx 13 bounce 4 command 1 flag 64 ctx.s.optical.y(ems) 4 
    2024-10-15 18:56:41.426  426037050 : ]./cxs_min.sh 

    
So there is something broken with curand usage in A. All of them are small when they should be uniform on 0->1::

    A[blyth@localhost CSGOptiX]$ VERSION=98 PIDX=4 ~/o/cxs_min.sh run | grep u_
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.00033755 logf(u_absorption) -7.99380875 absorption_length 41631.9062 absorption_distance 332797.500000 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.0003 u_absorption     0.0003 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.0005 TransCoeff     1.0000 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     1.0000 u_reflect     0.0005 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.00084386 logf(u_absorption) -7.07751799 absorption_length 41631.9062 absorption_distance 294650.562500 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.0008 u_absorption     0.0008 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.0010 TransCoeff     0.9570 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     0.9570 u_reflect     0.0010 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.00135018 logf(u_absorption) -6.60751486 absorption_length  1035.9432 absorption_distance 6845.010254 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.0013 u_absorption     0.0014 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.0015 TransCoeff     0.8992 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     0.8992 u_reflect     0.0015 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.00185650 logf(u_absorption) -6.28906107 absorption_length 41631.9062 absorption_distance 261825.593750 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.0018 u_absorption     0.0019 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.0020 TransCoeff     0.8757 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     0.8757 u_reflect     0.0020 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.00236282 logf(u_absorption) -6.04789925 absorption_length  1687.2012 absorption_distance 10204.022461 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.0023 u_absorption     0.0024 
    //qsim.propagate_at_surface_CustomART idx 4 lpmtid 1425 ARTE (   0.818   1.000   0.000   0.541 ) u_theAbsorption    0.002 action 1 
    //qsim.propagate_at_surface_CustomART.BREAK.SD/SA idx 4 lpmtid 1425 ARTE (   0.818   1.000   0.000   0.541 ) u_theEfficiency    0.003 theEfficiency   0.541 flag 64 
    A[blyth@localhost CSGOptiX]$ 
     

    P[blyth@localhost CSGOptiX]$ VERSION=98 PIDX=4 ~/o/cxs_min.sh run | grep u_
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.88969451 logf(u_absorption) -0.11687706 absorption_length 41631.9062 absorption_distance 4865.814941 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.1631 u_absorption     0.8897 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.2414 TransCoeff     1.0000 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     1.0000 u_reflect     0.2414 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.14787784 logf(u_absorption) -1.91136873 absorption_length 41631.9062 absorption_distance 79573.921875 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.0786 u_absorption     0.1479 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.4265 TransCoeff     0.9570 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     0.9570 u_reflect     0.4265 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.66764003 logf(u_absorption) -0.40400606 absorption_length  1035.9432 absorption_distance 418.527344 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.4095 u_absorption     0.6676 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.2769 TransCoeff     0.8992 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     0.8992 u_reflect     0.2769 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.04952160 logf(u_absorption) -3.00534630 absorption_length 41631.9062 absorption_distance 125118.296875 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.1460 u_absorption     0.0495 
    //qsim.propagate_at_boundary.body idx 4 : u_reflect     0.5336 TransCoeff     0.8757 reflect 0 
    //qsim.propagate_at_boundary.tail idx 4 : reflect 0 tir 0 TransCoeff     0.8757 u_reflect     0.5336 
    //qsim.propagate_to_boundary.head idx 4 : u_absorption 0.47640604 logf(u_absorption) -0.74148464 absorption_length  1687.2012 absorption_distance 1251.033813 
    //qsim.propagate_to_boundary.head idx 4 : u_scattering     0.9105 u_absorption     0.4764 
    //qsim.propagate_at_surface_CustomART idx 4 lpmtid 1425 ARTE (   0.818   1.000   0.000   0.541 ) u_theAbsorption    0.644 action 1 
    //qsim.propagate_at_surface_CustomART.BREAK.SD/SA idx 4 lpmtid 1425 ARTE (   0.818   1.000   0.000   0.541 ) u_theEfficiency    0.120 theEfficiency   0.541 flag 64 
    P[blyth@localhost CSGOptiX]$ 





curand_init
------------

::

    P[blyth@localhost opticks]$ opticks-f curand_init
    ./bin/oks.bash:  * https://devtalk.nvidia.com/default/topic/770325/optix/curand_init-within-optix/        Suggestion of insufficent stack 
    ./cudarap/cuRANDWrapper.cc:Performs multiple CUDA launches to curand_init
    ./cudarap/cuRANDWrapper_kernel.cu:Invokes curand_init with resulting curandState written into rng_states 
    ./cudarap/cuRANDWrapper_kernel.cu:including the curand_init one
    ./cudarap/cuRANDWrapper_kernel.cu:(On macOS) curand_init runs 10x slower for large thread_offset ? 
    ./cudarap/cuRANDWrapper_kernel.cu:* :google:`curand_init slow with large sequence numbers`
    ./cudarap/cuRANDWrapper_kernel.cu:    __device__ void curand_init (
    ./cudarap/cuRANDWrapper_kernel.cu:The curand_init() function sets up an initial state allocated by the caller using the
    ./cudarap/cuRANDWrapper_kernel.cu:    curand_init(seed, id + thread_offset , offset, &rng_states[id]);  
    ./cudarap/cudarap.bash:was loaded from cache as opposed to being curand_init::
    ./cudarap/tests/curand_aligned_device.cu:For the device API using curand_init(), you explicitly give the subsequence
    ./cudarap/tests/curand_aligned_device.cu:one call curand_init() with the same seed and subsequence numbers from 0 to
    ./cudarap/tests/curand_aligned_device.cu:     98    // including the curand_init one
    ./cudarap/tests/curand_aligned_device.cu:    113    curand_init(seed, id + thread_offset , offset, &rng_states[id]);
    ./cudarap/tests/curand_aligned_device.cu:    117    // curand_init runs 10x slower for large thread_offset ? starting from 262144
    ./cudarap/tests/curand_aligned_device.cu:    120    // :google:`curand_init slow with large sequence numbers`
    ./cudarap/tests/curand_aligned_device.cu:    curand_init(1234,0,0,&rngState);
    ./cudarap/tests/curand_aligned_device.cu:        //curand_init(1234,i,0,&rngState); // i: sequence number
    ./cudarap/tests/curand_aligned_host.cc:For the device API using curand_init(), you explicitly give the subsequence
    ./cudarap/tests/curand_aligned_host.cc:one call curand_init() with the same seed and subsequence numbers from 0 to
    ./cudarap/tests/curand_skipahead.cu:    curand_init(seed, id + thread_offset , offset, &rng_states[id]);  
    ./examples/UseCUDARapThrust/UseCUDARapThrust.cu:        curand_init(seed, 0, 0, &s); 
    ./externals/optixnote.bash:* https://devtalk.nvidia.com/default/topic/770325/curand_init-within-optix/
    ./externals/optixnote.bash:    144     curand_init(seed, id, offset, &s[id]);
    ./externals/optixnote.bash:    120     curand_init(seed, id, offset, &s[id]);
    ./externals/optixnote.bash:  for curand_init with subsequences and probably changing stack size 
    ./externals/optixnote.bash:  do curand_init and prepare the curandState buffer for interop
    ./notes/issues/ilker-hot-spots-reply.txt:One problem with using curand is that the curand_init initialization 
    ./notes/issues/ilker-hot-spots-reply.txt:The stack size needed to do curand_init is hugely more that the 
    ./notes/issues/ilker-hot-spots-reply.txt:Because of this Opticks does that curand_init for the configured maximum number 
    ./qudarap/QCurandState.cc:extern "C" void QCurandState_curand_init(SLaunchSequence* lseq, qcurandstate* cs, qcurandstate* d_cs) ; 
    ./qudarap/QCurandState.cc:    QCurandState_curand_init(lseq, cs, d_cs); 
    ./qudarap/QCurandState.cc:    LOG(info) << "after QCurandState_curand_init lseq.desc " << std::endl << lseq->desc() ; 
    ./qudarap/QCurandState.cu:__global__ void _QCurandState_curand_init(int threads_per_launch, int thread_offset, qcurandstate* cs, curandState* states_thread_offset )
    ./qudarap/QCurandState.cu:    curand_init(cs->seed, id+thread_offset, cs->offset, states_thread_offset + id );  
    ./qudarap/QCurandState.cu:    //if( id == 0 ) printf("// _QCurandState_curand_init thread_offset %d \n", thread_offset ); 
    ./qudarap/QCurandState.cu:extern "C" void QCurandState_curand_init(SLaunchSequence* seq,  qcurandstate* cs, qcurandstate* d_cs) 
    ./qudarap/QCurandState.cu:    printf("//QCurandState_curand_init seq.items %d cs %p  d_cs %p cs.num %llu \n", seq->items, cs, d_cs, cs->num );  
    ./qudarap/QCurandState.cu:        _QCurandState_curand_init<<<l.blocks_per_launch,l.threads_per_block>>>( l.threads_per_launch, l.thread_offset, d_cs, states_thread_offset  );  
    ./qudarap/QCurandState.hh:calling curand_init and they need to be downloaded and stored
    ./qudarap/QCurandState.hh:A difficulty is that calling curand_init is a very heavy kernel, 
    ./sysrap/tests/curand_uniform_test.cu:    curand_init( seed, subsequence, offset, &rng ); 
    ./thrustrap/TCURAND.hh:2. does the curand_init when could use the persisted curandState files
    ./thrustrap/TRngBuf_.cu:Suspect the repeated curand_init for every id maybe a very 
    ./thrustrap/TRngBuf_.cu:    curand_init(m_seed, m_ibase + uid , m_offset, &s); 
    ./thrustrap/tests/rng.cu:        curand_init(seed, uid , offset, &s);
    ./thrustrap/tests/thrust_curand_estimate_pi.cu:        curand_init(seed, 0, 0, &rng); 
    ./thrustrap/tests/thrust_curand_printf.cu:        curand_init(_seed, id + thread_offset, _offset, &s); 
    ./thrustrap/tests/thrust_curand_printf.cu:curand_init (
    ./thrustrap/tests/thrust_curand_printf.cu:The curand_init() function sets up an initial state allocated by the caller
    ./thrustrap/tests/thrust_curand_printf_redirect.cu:        curand_init(_seed, id + thread_offset, _offset, &s); 
    ./thrustrap/tests/thrust_curand_printf_redirect.cu:curand_init (
    ./thrustrap/tests/thrust_curand_printf_redirect.cu:The curand_init() function sets up an initial state allocated by the caller
    ./thrustrap/tests/thrust_curand_printf_redirect2.cu:        curand_init(_seed, id + thread_offset, _offset, &s); 
    ./thrustrap/tests/thrust_curand_printf_redirect2.cu:curand_init (
    ./thrustrap/tests/thrust_curand_printf_redirect2.cu:The curand_init() function sets up an initial state allocated by the caller
    P[blyth@localhost opticks]$ 


Add simple test of curand_uniform that does its own curand_init::

    sysrap/tests/curand_uniform_test.cu
    sysrap/tests/curand_uniform_test.py
    sysrap/tests/curand_uniform_test.sh

