FIXED cxs_min_major_difference_between_TITAN_RTX_OptiX7p5_and_RTX_5000_Ada_OptiX8p0
======================================================================================


Issue : major simulation differences between P and A (different CUDA and OptiX and GPU)
-----------------------------------------------------------------------------------------

Comparing:

* P : NVIDIA TITAN RTX, OptiX 7.5
* A : NVIDIA RTX 5000 Ada Generation, OptiX 8.0


Current hypothesis of whats gone wrong
----------------------------------------

HMM. Likely the issue in incompatibility between curandState between curand versions. 

* thats still a possibility : but find the issue is fixed 
  by recreation of the QCurandState files


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




curand_uniform_test with live curand_init : gives same randoms
----------------------------------------------------------------

Add simple test of curand_uniform that does its own curand_init::

    sysrap/tests/curand_uniform_test.cu
    sysrap/tests/curand_uniform_test.py
    sysrap/tests/curand_uniform_test.sh


::

    A[blyth@localhost opticks]$ sysrap/tests/curand_uniform_test.sh ana
    a.shape
     (1000, 16)
    a[:10]
     [[0.74022 0.43845 0.51701 0.15699 0.07137 0.46251 0.22764 0.32936 0.14407 0.1878  0.91538 0.54012 0.97466 0.54747 0.65316 0.23024]
     [0.92099 0.46036 0.33346 0.37252 0.4896  0.56727 0.07991 0.23337 0.50938 0.08898 0.00671 0.95423 0.54671 0.82455 0.52706 0.93013]
     [0.03902 0.25021 0.18448 0.96242 0.52055 0.93996 0.83058 0.40973 0.08162 0.80677 0.69529 0.61771 0.25633 0.21368 0.34242 0.22408]
     [0.96896 0.49474 0.67338 0.56277 0.12019 0.97649 0.13583 0.58897 0.49062 0.32844 0.91143 0.19068 0.9637  0.89755 0.62429 0.71015]
     [0.92514 0.05301 0.1631  0.88969 0.56664 0.24142 0.49369 0.32123 0.07861 0.14788 0.59866 0.42647 0.24347 0.48918 0.40953 0.66764]
     [0.44635 0.3377  0.20723 0.98454 0.40279 0.1781  0.45992 0.16001 0.36089 0.62038 0.45004 0.30574 0.50284 0.45595 0.5516  0.84838]
     [0.66732 0.39676 0.15829 0.5423  0.7056  0.12585 0.15365 0.65258 0.37992 0.85478 0.20781 0.0901  0.70118 0.43362 0.10571 0.08183]
     [0.10993 0.87442 0.98075 0.96693 0.16233 0.42767 0.93141 0.01003 0.84566 0.37989 0.81176 0.15237 0.27327 0.41338 0.78616 0.08703]
     [0.47022 0.48217 0.42791 0.44174 0.78041 0.85861 0.61435 0.80234 0.65919 0.59214 0.18296 0.71884 0.92713 0.42197 0.01055 0.82696]
     [0.51319 0.04284 0.95184 0.92588 0.25979 0.91341 0.39325 0.83318 0.27532 0.75222 0.66639 0.03765 0.87857 0.96512 0.03355 0.81466]]
    a[-10:]
     [[0.41888 0.56394 0.26219 0.00544 0.34131 0.24802 0.02585 0.42882 0.45842 0.68441 0.1162  0.07948 0.70902 0.93657 0.54654 0.41797]
     [0.6406  0.80706 0.12232 0.20049 0.90991 0.13225 0.18421 0.27288 0.83271 0.89976 0.48249 0.51084 0.22823 0.63753 0.43524 0.96682]
     [0.29197 0.19001 0.98212 0.68296 0.65355 0.74176 0.84946 0.58338 0.30676 0.91659 0.78078 0.0342  0.73427 0.05188 0.61055 0.85   ]
     [0.84055 0.33497 0.81023 0.68106 0.82873 0.87127 0.75434 0.55597 0.85694 0.36502 0.91378 0.68908 0.53978 0.20404 0.01672 0.14249]
     [0.6201  0.62216 0.83531 0.72095 0.70984 0.75301 0.60597 0.11183 0.2665  0.62516 0.12829 0.27882 0.71579 0.59997 0.41287 0.72082]
     [0.42809 0.7106  0.64159 0.94931 0.23182 0.09769 0.12973 0.39439 0.7484  0.05785 0.79519 0.12628 0.15853 0.12913 0.14954 0.98629]
     [0.93038 0.01259 0.53405 0.20617 0.06964 0.78301 0.62946 0.97189 0.22707 0.7842  0.72258 0.9895  0.12467 0.85368 0.76313 0.08281]
     [0.15602 0.99039 0.6817  0.11667 0.13779 0.3867  0.73269 0.66636 0.00007 0.97589 0.64677 0.22477 0.44537 0.20699 0.73511 0.35352]
     [0.16356 0.4678  0.83821 0.44082 0.21579 0.71205 0.03324 0.69551 0.22208 0.92826 0.24047 0.18735 0.79577 0.88763 0.34437 0.94503]
     [0.21777 0.24313 0.72559 0.24963 0.08471 0.51074 0.23489 0.12473 0.75238 0.91716 0.68549 0.11767 0.76911 0.00663 0.21612 0.32016]]
    A[blyth@localhost opticks]$ 


    P[blyth@localhost opticks]$ sysrap/tests/curand_uniform_test.sh ana
    a.shape
     (1000, 16)
    a[:10]
     [[0.74022 0.43845 0.51701 0.15699 0.07137 0.46251 0.22764 0.32936 0.14407 0.1878  0.91538 0.54012 0.97466 0.54747 0.65316 0.23024]
     [0.92099 0.46036 0.33346 0.37252 0.4896  0.56727 0.07991 0.23337 0.50938 0.08898 0.00671 0.95423 0.54671 0.82455 0.52706 0.93013]
     [0.03902 0.25021 0.18448 0.96242 0.52055 0.93996 0.83058 0.40973 0.08162 0.80677 0.69529 0.61771 0.25633 0.21368 0.34242 0.22408]
     [0.96896 0.49474 0.67338 0.56277 0.12019 0.97649 0.13583 0.58897 0.49062 0.32844 0.91143 0.19068 0.9637  0.89755 0.62429 0.71015]
     [0.92514 0.05301 0.1631  0.88969 0.56664 0.24142 0.49369 0.32123 0.07861 0.14788 0.59866 0.42647 0.24347 0.48918 0.40953 0.66764]
     [0.44635 0.3377  0.20723 0.98454 0.40279 0.1781  0.45992 0.16001 0.36089 0.62038 0.45004 0.30574 0.50284 0.45595 0.5516  0.84838]
     [0.66732 0.39676 0.15829 0.5423  0.7056  0.12585 0.15365 0.65258 0.37992 0.85478 0.20781 0.0901  0.70118 0.43362 0.10571 0.08183]
     [0.10993 0.87442 0.98075 0.96693 0.16233 0.42767 0.93141 0.01003 0.84566 0.37989 0.81176 0.15237 0.27327 0.41338 0.78616 0.08703]
     [0.47022 0.48217 0.42791 0.44174 0.78041 0.85861 0.61435 0.80234 0.65919 0.59214 0.18296 0.71884 0.92713 0.42197 0.01055 0.82696]
     [0.51319 0.04284 0.95184 0.92588 0.25979 0.91341 0.39325 0.83318 0.27532 0.75222 0.66639 0.03765 0.87857 0.96512 0.03355 0.81466]]
    a[-10:]
     [[0.41888 0.56394 0.26219 0.00544 0.34131 0.24802 0.02585 0.42882 0.45842 0.68441 0.1162  0.07948 0.70902 0.93657 0.54654 0.41797]
     [0.6406  0.80706 0.12232 0.20049 0.90991 0.13225 0.18421 0.27288 0.83271 0.89976 0.48249 0.51084 0.22823 0.63753 0.43524 0.96682]
     [0.29197 0.19001 0.98212 0.68296 0.65355 0.74176 0.84946 0.58338 0.30676 0.91659 0.78078 0.0342  0.73427 0.05188 0.61055 0.85   ]
     [0.84055 0.33497 0.81023 0.68106 0.82873 0.87127 0.75434 0.55597 0.85694 0.36502 0.91378 0.68908 0.53978 0.20404 0.01672 0.14249]
     [0.6201  0.62216 0.83531 0.72095 0.70984 0.75301 0.60597 0.11183 0.2665  0.62516 0.12829 0.27882 0.71579 0.59997 0.41287 0.72082]
     [0.42809 0.7106  0.64159 0.94931 0.23182 0.09769 0.12973 0.39439 0.7484  0.05785 0.79519 0.12628 0.15853 0.12913 0.14954 0.98629]
     [0.93038 0.01259 0.53405 0.20617 0.06964 0.78301 0.62946 0.97189 0.22707 0.7842  0.72258 0.9895  0.12467 0.85368 0.76313 0.08281]
     [0.15602 0.99039 0.6817  0.11667 0.13779 0.3867  0.73269 0.66636 0.00007 0.97589 0.64677 0.22477 0.44537 0.20699 0.73511 0.35352]
     [0.16356 0.4678  0.83821 0.44082 0.21579 0.71205 0.03324 0.69551 0.22208 0.92826 0.24047 0.18735 0.79577 0.88763 0.34437 0.94503]
     [0.21777 0.24313 0.72559 0.24963 0.08471 0.51074 0.23489 0.12473 0.75238 0.91716 0.68549 0.11767 0.76911 0.00663 0.21612 0.32016]]
    P[blyth@localhost opticks]$ 



QRngTest.sh  : YEP thats messed up on A
------------------------------------------

::

    P[blyth@localhost tests]$  ~/o/qudarap/tests/QRngTest.sh
                    FOLD : /data/blyth/opticks/QRngTest
                     bin : QRngTest
                  script : QRngTest.py
    2024-10-15 20:39:54.363 INFO  [279233] [main@102] QRng path /home/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin rngmax 3000000 qr 0x699530 qr.skipahead_event_offset 1 d_qr 0x7fc07aa00000
    //QRng_generate_2 event_idx 0 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 1 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 2 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 3 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 4 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 5 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 6 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 7 ni 100 nv 256 
    2024-10-15 20:39:54.367 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 8 ni 100 nv 256 
    2024-10-15 20:39:54.368 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 9 ni 100 nv 256 
    2024-10-15 20:39:54.368 INFO  [279233] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    2024-10-15 20:39:54.369 INFO  [279233] [test_generate_2@88] save to /data/blyth/opticks/QRngTest/float
    uu.shape
     (10, 100, 256)
    uu[:10]
     [[[0.74022 0.43845 0.51701 0.15699 ... 0.07978 0.59805 0.81959 0.14472]
      [0.92099 0.46036 0.33346 0.37252 ... 0.24695 0.90173 0.45439 0.58697]
      [0.03902 0.25021 0.18448 0.96242 ... 0.21389 0.52502 0.02501 0.47301]
      [0.96896 0.49474 0.67338 0.56277 ... 0.44728 0.60353 0.25211 0.45708]
      ...



    A[blyth@localhost opticks]$ qudarap/tests/QRngTest.sh 
                    FOLD : /data1/blyth/tmp/QRngTest
                     bin : QRngTest
                  script : QRngTest.py
    2024-10-15 20:42:08.895 INFO  [124163] [main@102] QRng path /home/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin rngmax 3000000 qr 0x7ea6a0 qr.skipahead_event_offset 1 d_qr 0x7fa242a00000
    //QRng_generate_2 event_idx 0 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 1 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 2 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 3 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 4 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 5 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 6 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 7 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 8 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 9 ni 100 nv 256 
    2024-10-15 20:42:08.898 INFO  [124163] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    2024-10-15 20:42:08.899 INFO  [124163] [test_generate_2@88] save to /data1/blyth/tmp/QRngTest/float
    uu.shape
     (10, 100, 256)
    uu[:10]
     [[[0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]
      [0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]
      [0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]
      [0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]
      ...
      [0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]
      [0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]
      [0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]
      [0.00008 0.00017 0.00025 0.00034 ... 0.02135 0.02143 0.02152 0.0216 ]]

     [[0.00017 0.00025 0.00034 0.00042 ... 0.02143 0.02152 0.0216  0.02169]
      [0.00017 0.00025 0.00034 0.00042 ... 0.02143 0.02152 0.0216  0.02169]
      [0.00017 0.00025 0.00034 0.00042 ... 0.02143 0.02152 0.0216  0.02169]
      [0.00017 0.00025 0.00034 0.00042 ... 0.02143 0.02152 0.0216  0.02169]
      ...



HMM: Mystified : on A moving the curandState aside and recreating fixes the issue
-----------------------------------------------------------------------------------

::

    A[blyth@localhost tests]$ l ~/.opticks/rngcache/RNG/
    total 601568
    429688 -rw-r--r--. 1 blyth blyth 440000000 Aug 29 17:17 QCurandState_10000000_0_0.bin
         0 drwxr-xr-x. 2 blyth blyth       115 Aug 29 17:17 .
    128908 -rw-r--r--. 1 blyth blyth 132000000 Aug 29 17:17 QCurandState_3000000_0_0.bin
     42972 -rw-r--r--. 1 blyth blyth  44000000 Aug 29 17:17 QCurandState_1000000_0_0.bin
         0 drwxr-xr-x. 3 blyth blyth        17 Aug 29 17:17 ..
    A[blyth@localhost tests]$ cd ~/.opticks/rngcache/
    A[blyth@localhost rngcache]$ mv RNG RNG.old
    A[blyth@localhost rngcache]$ 
    A[blyth@localhost rngcache]$ 
    A[blyth@localhost rngcache]$ qudarap-
    A[blyth@localhost rngcache]$ t qudarap-prepare-installation
    qudarap-prepare-installation () 
    { 
        local sizes=$(qudarap-prepare-sizes);
        local size;
        local seed=${QUDARAP_RNG_SEED:-0};
        local offset=${QUDARAP_RNG_OFFSET:-0};
        for size in $sizes;
        do
            QCurandState_SPEC=$size:$seed:$offset ${OPTICKS_PREFIX}/lib/QCurandStateTest;
            rc=$?;
            [ $rc -ne 0 ] && return $rc;
        done;
        return 0
    }
    A[blyth@localhost rngcache]$ qudarap-prepare-installation


    A[blyth@localhost tests]$ ./QRngTest.sh 
                    FOLD : /data1/blyth/tmp/QRngTest
                     bin : QRngTest
                  script : QRngTest.py
    2024-10-15 20:53:12.164 INFO  [124799] [QRng::init@48] [QRng__init_VERBOSE] YES
    QRng path /home/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin rngmax 3000000 qr 0x1d766a0 qr.skipahead_event_offset 1 d_qr 0x7fae0aa00000
    2024-10-15 20:53:12.164 INFO  [124799] [main@102] QRng path /home/blyth/.opticks/rngcache/RNG/QCurandState_3000000_0_0.bin rngmax 3000000 qr 0x1d766a0 qr.skipahead_event_offset 1 d_qr 0x7fae0aa00000
    //QRng_generate_2 event_idx 0 ni 100 nv 256 
    2024-10-15 20:53:12.166 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 1 ni 100 nv 256 
    2024-10-15 20:53:12.166 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 2 ni 100 nv 256 
    2024-10-15 20:53:12.166 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 3 ni 100 nv 256 
    2024-10-15 20:53:12.166 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 4 ni 100 nv 256 
    2024-10-15 20:53:12.167 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 5 ni 100 nv 256 
    2024-10-15 20:53:12.167 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 6 ni 100 nv 256 
    2024-10-15 20:53:12.167 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 7 ni 100 nv 256 
    2024-10-15 20:53:12.167 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 8 ni 100 nv 256 
    2024-10-15 20:53:12.167 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    //QRng_generate_2 event_idx 9 ni 100 nv 256 
    2024-10-15 20:53:12.167 INFO  [124799] [QU::copy_device_to_host_and_free@462] copy 25600 sizeof(T) 4 label QRng::generate_2:ni*nv
    2024-10-15 20:53:12.168 INFO  [124799] [test_generate_2@88] save to /data1/blyth/tmp/QRngTest/float
    uu.shape
     (10, 100, 256)
    uu[:10]
     [[[0.74022 0.43845 0.51701 0.15699 ... 0.07978 0.59805 0.81959 0.14472]
      [0.92099 0.46036 0.33346 0.37252 ... 0.24695 0.90173 0.45439 0.58697]
      [0.03902 0.25021 0.18448 0.96242 ... 0.21389 0.52502 0.02501 0.47301]
      [0.96896 0.49474 0.67338 0.56277 ... 0.44728 0.60353 0.25211 0.45708]
      ...
      [0.30224 0.78633 0.26038 0.86015 ... 0.3562  0.67672 0.35955 0.02354]
      [0.80768 0.26517 0.98403 0.40043 ... 0.54698 0.55139 0.98299 0.85286]
      [0.40713 0.28182 0.36872 0.77379 ... 0.01637 0.36403 0.48313 0.05647]
      [0.75132 0.35347 0.88852 0.08289 ... 0.18814 0.75153 0.48603 0.35428]]



