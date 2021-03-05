raja_repeated_photons
=======================


Raja
-----

::

    Hi Simon,

    Thanks to glibc being fixed on our machines, the code has compiled (multiple
    times!) and run CerenkovMinimal quite happily. Many thanks for this - I have
    even managed to get a fairly straightforward guide to installing it on our
    machines too which is quite useful.

    Now for the problem. CerenkovMinimal is running fine, iterating over multiple
    events. Looking at the dumps, the photons from Geant4 are also nicely produced
    and quite random. However, the photons produced by opticks are repeated for
    every event. The total number of photons does change at random (set in Geant?)
    but the actual position and momentum of the photons is the same each time.

    This suggests to me that the random number generator in opticks is being reset
    at the end of each event. Would you by chance be able to point me to where this
    is happening? Or is it something else altogether? I tried to find out how to
    increase the loglevel, but it was not obvious to me (I had no increase in
    logging, whatever environment variable I set, looking at the documentation) -
    here I am guessing that I need to do it for thrustrap.

    Thanks a lot again and Kind Regards, Raja.





Added skipahead handling
---------------------------

* --skipaheadstep 

* m_skipahead in OpticksEvent 
* passed into GPU context by OPropagator/ORng 

* problems with this approach
  
1. need to guess a step for how many randoms any event will consume, 
   or just pick a large step (once have moved to ULL)

TODO: 

* check reproducibility with masked running where pick 
  a selection of events and run just those 

* move to unsigned long long skipahead (probably via uint2 gymnastics
  to get it into the GPU context)



::

    Changes to be committed:
      (use "git reset HEAD <file>..." to unstage)

        new file:   cudarap/tests/NP.hh
        new file:   cudarap/tests/NPU.hh
        new file:   cudarap/tests/curand_skipahead.cu
        new file:   cudarap/tests/curand_skipahead.py
        modified:   examples/Geant4/CerenkovMinimal/src/EventAction.cc
        modified:   examples/Geant4/CerenkovMinimal/src/RunAction.cc
        modified:   examples/UseGeant4/UseGeant4.cc
        new file:   notes/issues/raja_repeated_photons.rst
        modified:   optickscore/Opticks.cc
        modified:   optickscore/Opticks.hh
        modified:   optickscore/OpticksCfg.cc
        modified:   optickscore/OpticksCfg.hh
        modified:   optickscore/OpticksEvent.cc
        modified:   optickscore/OpticksEvent.hh
        modified:   optickscore/OpticksRun.cc
        modified:   optixrap/OPropagator.cc
        modified:   optixrap/OPropagator.hh
        modified:   optixrap/ORng.cc
        modified:   optixrap/ORng.hh
        modified:   optixrap/cu/generate.cu



    epsilon:opticks blyth$ opticks-f SkipAhead
    ./optickscore/OpticksEvent.hh:       void     setSkipAhead(unsigned skipahead);
    ./optickscore/OpticksEvent.hh:       unsigned getSkipAhead() const ;
    ./optickscore/OpticksEvent.cc:void OpticksEvent::setSkipAhead(unsigned skipahead)  // TODO: move to unsigned long long 
    ./optickscore/OpticksEvent.cc:unsigned OpticksEvent::getSkipAhead() const 
    ./optickscore/Opticks.hh:       unsigned             getSkipAheadStep() const ;  // --skipaheadstep 1000
    ./optickscore/OpticksCfg.cc:unsigned OpticksCfg<Listener>::getSkipAheadStep() const 
    ./optickscore/OpticksRun.cc:    unsigned skipaheadstep = m_ok->getSkipAheadStep() ; 
    ./optickscore/OpticksRun.cc:    m_evt->setSkipAhead( skipahead ); // TODO: make configurable + move to ULL
    ./optickscore/OpticksCfg.hh:     unsigned     getSkipAheadStep() const ;
    ./optickscore/Opticks.cc:unsigned Opticks::getSkipAheadStep() const  // --skipaheadstep 1000
    ./optickscore/Opticks.cc:    return m_cfg->getSkipAheadStep();
    ./optixrap/OPropagator.hh:        void     setSkipAhead(unsigned skipahead);
    ./optixrap/OPropagator.hh:        unsigned getSkipAhead() const ;
    ./optixrap/OPropagator.cc:void OPropagator::setSkipAhead(unsigned skipahead)
    ./optixrap/OPropagator.cc:    m_orng->setSkipAhead(skipahead); 
    ./optixrap/OPropagator.cc:unsigned OPropagator::getSkipAhead() const 
    ./optixrap/OPropagator.cc:   return m_orng->getSkipAhead();  
    ./optixrap/OPropagator.cc:    unsigned skipahead = evt->getSkipAhead(); 
    ./optixrap/OPropagator.cc:    setSkipAhead(skipahead);     
    ./optixrap/ORng.hh:      void setSkipAhead(unsigned skipahead); 
    ./optixrap/ORng.hh:      unsigned getSkipAhead() const ;
    ./optixrap/ORng.cc:void ORng::setSkipAhead( unsigned skipahead )
    ./optixrap/ORng.cc:unsigned ORng::getSkipAhead() const 
    epsilon:opticks blyth$ opticks-f skipahead
    ./cudarap/tests/curand_skipahead.py:    a = np.load("/tmp/curand_skipahead_1.npy")
    ./cudarap/tests/curand_skipahead.py:    b = np.load("/tmp/curand_skipahead_2.npy")
    ./cudarap/tests/curand_skipahead.cu:// nvcc curand_skipahead.cu -std=c++11 -ccbin=/usr/bin/clang -o /tmp/curand_skipahead && /tmp/curand_skipahead 
    ./cudarap/tests/curand_skipahead.cu:curand_skipahead.cu
    ./cudarap/tests/curand_skipahead.cu:    skipahead( skip, &rng_states[id]) ;
    ./cudarap/tests/curand_skipahead.cu:        skipahead( skip, &rng_states[id]) ;
    ./cudarap/tests/curand_skipahead.cu:    ss << "/tmp/curand_skipahead_" << mode << ".npy" ;
    ./bin/curand.bash:skipahead
    ./optickscore/OpticksEvent.hh:       void     setSkipAhead(unsigned skipahead);
    ./optickscore/OpticksEvent.hh:       unsigned        m_skipahead ; 
    ./optickscore/OpticksEvent.cc:    m_skipahead(0)
    ./optickscore/OpticksEvent.cc:void OpticksEvent::setSkipAhead(unsigned skipahead)  // TODO: move to unsigned long long 
    ./optickscore/OpticksEvent.cc:    m_skipahead = skipahead ; 
    ./optickscore/OpticksEvent.cc:    return m_skipahead ; 
    ./optickscore/Opticks.hh:       unsigned             getSkipAheadStep() const ;  // --skipaheadstep 1000
    ./optickscore/OpticksCfg.cc:    m_skipaheadstep(0),     
    ./optickscore/OpticksCfg.cc:   char skipaheadstep[128];
    ./optickscore/OpticksCfg.cc:   snprintf(skipaheadstep,128, 
    ./optickscore/OpticksCfg.cc:"Unsigned int skipaheadstep used with ORng"
    ./optickscore/OpticksCfg.cc:"Default %d ", m_skipaheadstep);
    ./optickscore/OpticksCfg.cc:       ("skipaheadstep",  boost::program_options::value<unsigned>(&m_skipaheadstep), skipaheadstep );
    ./optickscore/OpticksCfg.cc:    return m_skipaheadstep ; 
    ./optickscore/OpticksRun.cc:    unsigned skipaheadstep = m_ok->getSkipAheadStep() ; 
    ./optickscore/OpticksRun.cc:    unsigned skipahead =  tagoffset*skipaheadstep ; 
    ./optickscore/OpticksRun.cc:        << " skipaheadstep " << skipaheadstep
    ./optickscore/OpticksRun.cc:        << " skipahead " << skipahead
    ./optickscore/OpticksRun.cc:    m_evt->setSkipAhead( skipahead ); // TODO: make configurable + move to ULL
    ./optickscore/OpticksCfg.hh:     unsigned    m_skipaheadstep ; 
    ./optickscore/Opticks.cc:unsigned Opticks::getSkipAheadStep() const  // --skipaheadstep 1000
    ./examples/Geant4/CerenkovMinimal/src/RunAction.cc:    const char* embedded_commandline_extra = "--skipaheadstep 1000" ; // see ~/opticks/notes/issues/raja_repeated_photons.rst 
    ./optixrap/OPropagator.hh:        void     setSkipAhead(unsigned skipahead);
    ./optixrap/cu/generate.cu:// rng_states rng_skipahead
    ./optixrap/cu/generate.cu:    unsigned long long rng_skipahead_ = rng_skipahead ;   // see ORng.hh
    ./optixrap/cu/generate.cu:    if( rng_skipahead_ > 0ull )
    ./optixrap/cu/generate.cu:        skipahead(rng_skipahead_ , &rng) ;
    ./optixrap/OPropagator.cc:void OPropagator::setSkipAhead(unsigned skipahead)
    ./optixrap/OPropagator.cc:    m_orng->setSkipAhead(skipahead); 
    ./optixrap/OPropagator.cc:    unsigned skipahead = evt->getSkipAhead(); 
    ./optixrap/OPropagator.cc:    LOG(info)  << " skipahead " << skipahead ;  
    ./optixrap/OPropagator.cc:    LOG(LEVEL) << " skipahead " << skipahead ;  
    ./optixrap/OPropagator.cc:    setSkipAhead(skipahead);     
    ./optixrap/ORng.hh:rtDeclareVariable(unsigned int,  rng_skipahead, , );
    ./optixrap/ORng.hh://rtDeclareVariable(unsigned long long,  rng_skipahead, , );
    ./optixrap/ORng.hh:      void setSkipAhead(unsigned skipahead); 
    ./optixrap/ORng.hh:      unsigned        m_rng_skipahead ;   
    ./optixrap/ORng.cc:    m_rng_skipahead(0)   
    ./optixrap/ORng.cc:    m_context["rng_skipahead"]->setUint(m_rng_skipahead) ; 
    ./optixrap/ORng.cc:void ORng::setSkipAhead( unsigned skipahead )
    ./optixrap/ORng.cc:    LOG(LEVEL) << " skipahead " << skipahead ; 
    ./optixrap/ORng.cc:    m_rng_skipahead = skipahead ; 
    ./optixrap/ORng.cc:    m_context["rng_skipahead"]->setUint(m_rng_skipahead) ; 
    ./optixrap/ORng.cc:    return m_rng_skipahead ; 
    epsilon:opticks blyth$ 


