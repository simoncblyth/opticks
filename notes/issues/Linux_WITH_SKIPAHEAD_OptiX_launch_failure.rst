Linux_WITH_SKIPAHEAD_OptiX_launch_failure
================================================

> I will have a look at the gensteps. I would have
> thought that even if I put in the exact gensteps each time it would be ok. As
> long as the random numbers used to generate the optical photons are different
> and so wavelength direction etc should vary accordingly. 

Actually, I recall seeing something like this arising from almost equal gensteps with
g4ok/tests/G4OKTest.cc. From a comment in that test:

    Identical Photons from Near Identical Gensteps Issue
    -------------------------------------------------------

    Because this test uses artifical and almost identical "torch" gensteps that 
    differ only in the number of photons this will generate duplicated photons 
    for each "event",

    In future using curand skipahead WITH_SKIPAHEAD will allow the duplication to be avoided 
    but anyhow it is useful to not randomize by default as it then makes problems 
    of repeated gensteps easier to notice.  


The first thing is to understand why the gensteps are so similar. 
If they turn out to be correct then you can experiment with using 
the below option which switches on the use of curand skipahead to
define the starting point offsets in the random streams for every photon 
of each event.  

    --skipaheadstep 10000

The curand skipahead gets set to: tagoffset*skipaheadstep 
where the tagoffset is a 0-based event index.

The skipaheadstep value should be more than the maximum number of randoms consumed 
for the generation and propagation of any photon of the event. However
the distribution of the number of randoms consumed is typically peaked
at perhaps a few hundred with a long tail, so it would be difficult 
to notice problems from reusing randoms as so few of them would be reused 
if you use a reasonable skipaheadstep.    

This "absolute" approach is taken in order to allow any photon 
of any event to be reproduced in isolation without any need to 
store curandstate following runs. 

Another motivation for the curand usage approach with fixed curandState 
that gets initialized at installation and loaded from file prior to simulation is that 
the curand initialization is very stack expensive so you cannot get good ray tracing 
performance when you initialize curand and generate/simulate
in the same kernel call because of the huge stack required for the curand
initialization. 

Unfortunately I have found some issues when using skipahead in 
Linux, OptiX 6.5, CUDA 10.1 launches but not with macOS, OptiX 5,CUDA 9.1. 

Specifically I get an uninformative OptiX launch failure, for OKTest and others tests::

    2021-08-15 22:26:59.768 FATAL [340052] [ORng::setSkipAhead@156] WITH_SKIPAHEAD skipahead 0
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)

Because of this issue WITH_SKIPAHEAD from optickscore/OpticksSwitches.h
is not enabled and the default *skipaheadstep* is zero.

Next week I plan to investigate skipahead in the qudarap pure CUDA context, 
by attempting to create a minimal reproducer that I can report to NVIDIA. 
"qudarap" is a pure CUDA opticks sub-package I am using to develop a more modular way of generating and simulating 
with much finer grained testing and greater control over the CUDA context.  This 
is in preparation for the migration to OptiX 7.

If you want to try skipahead to see if this problem effects you too, first switch on WITH_SKIPAHEAD:: 

    cd ~/opticks
    vi optickscors/OpticksSwitches.h
    om-cleaninstall     ## clean install is advisable after changing switches 

As there may be variation with CUDA versions, please report problems in detail 
if you run into anything.

::

    epsilon:opticks blyth$ opticks-f WITH_SKIPAHEAD
    ./g4ok/tests/G4OKTest.cc:In future using curand skipahead WITH_SKIPAHEAD will allow the duplication to be avoided 
    ./optickscore/OpticksSwitches.h://#define WITH_SKIPAHEAD 1
    ./optickscore/OpticksSwitches.h:#ifdef WITH_SKIPAHEAD
    ./optickscore/OpticksSwitches.h:    ss << "WITH_SKIPAHEAD" ;   
    ./optixrap/cu/generate.cu:#ifdef WITH_SKIPAHEAD
    ./optixrap/ORng.hh:#ifdef WITH_SKIPAHEAD
    ./optixrap/ORng.cc:#ifdef WITH_SKIPAHEAD
    ./optixrap/ORng.cc:    LOG(fatal) << "WITH_SKIPAHEAD skipahead " << skipahead ; 
    ./optixrap/ORng.cc:    LOG(LEVEL) << " skip as as WITH_SKIPAHEAD not enabled " ; 
    epsilon:opticks blyth$ 


The crucial things to understand how skipahead is being used are, optickscore/OpticksRun.cc::

    151 OpticksEvent* OpticksRun::createOKEvent(unsigned tagoffset)
    152 {
    153     bool is_ok_event = true ;
    154     OpticksEvent* evt = m_ok->makeEvent(is_ok_event, tagoffset) ;
    155 
    156     unsigned skipaheadstep = m_ok->getSkipAheadStep() ;
    157     unsigned skipahead =  tagoffset*skipaheadstep ;
    158     LOG(info)
    159         << " tagoffset " << tagoffset
    160         << " skipaheadstep " << skipaheadstep
    161         << " skipahead " << skipahead
    162         ;
    163 
    164     evt->setSkipAhead( skipahead ); // TODO: make configurable + move to ULL
    165     return evt ;
    166 }

And in optixrap/cu/generate.cu::

    584 #ifdef WITH_SKIPAHEAD
    585     unsigned long long rng_skipahead_ = rng_skipahead ;   // see ORng.hh
    586     if( rng_skipahead_ > 0ull )
    587     {
    588         skipahead(rng_skipahead_ , &rng) ;
    589     }
    590 #endif


To understand in more detail "opticks-f skipahead"


> Weather here has been very extreme the last few days with very violent storms.
> I only just now got my network back after power outages last night. 
> concerning the meeting I will see that I find a time that fits everyone. 

OK. Any work day within 09:00-18:00 UK time is OK with me this week.

Simon



Try simplification by minimizing WITH_SKIPAHEAD differences
--------------------------------------------------------------


::


    581 
    582     curandState rng = rng_states[photon_id];
    583     
    584     //unsigned long long rng_skipahead_ = rng_skipahead ;   // see ORng.hh
    585     //unsigned long long rng_skipahead_ = 10ull ; 
    586     //rtPrintf("// rng_skipahead %d  %llu \n", rng_skipahead, rng_skipahead_); 
    587     //skipahead(rng_skipahead_ , &rng) ;  
    588     // ^^^^^^^^ see notes/issues/Linux_WITH_SKIPAHEAD_OptiX_launch_failure.rst
    589     



Seems no way round it.  Any use of skipahead with optix launch
is giving optix::Exception::


    2021-08-16 23:18:22.739 FATAL [412810] [ORng::setSkipAhead@155] skipahead 0
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Aborted (core dumped)



BUT have seen no such problems with pure CUDA quadarap QRngTest.



Next : prototype using qudarap with csgoptix 
------------------------------------------------
 

