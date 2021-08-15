skipahead_review
==================


curand skipahead
-------------------

The skipahead advances the random stream for each photon_id 



WITH_SKIPAHEAD
----------------

Recall a problem WITH_SKIPAHEAD but unable to reproduce it 
or find notes about it.

::

    epsilon:opticks blyth$ opticks-f WITH_SKIPAHEAD
    ./g4ok/tests/G4OKTest.cc:In future using curand skipahead WITH_SKIPAHEAD will allow the duplication to be avoided 
    ./optickscore/OpticksSwitches.h:#define WITH_SKIPAHEAD 1
    ./optickscore/OpticksSwitches.h:#ifdef WITH_SKIPAHEAD
    ./optickscore/OpticksSwitches.h:    ss << "WITH_SKIPAHEAD" ;   
    ./optixrap/cu/generate.cu:#ifdef WITH_SKIPAHEAD
    ./optixrap/ORng.hh:#ifdef WITH_SKIPAHEAD
    ./optixrap/ORng.cc:#ifdef WITH_SKIPAHEAD
    ./optixrap/ORng.cc:    LOG(fatal) << "WITH_SKIPAHEAD skipahead " << skipahead ; 
    ./optixrap/ORng.cc:    LOG(LEVEL) << " skip as as WITH_SKIPAHEAD not enabled " ; 
    epsilon:opticks blyth$ 


optixrap/ORng.hh::

    024 #ifdef __CUDACC__
     25 
     26 rtBuffer<curandState, 1>         rng_states ;
     27 #ifdef WITH_SKIPAHEAD
     28 rtDeclareVariable(unsigned int,  rng_skipahead, , );
     29 // TODO: pack ull into uint2 ? as this number can get real big 
     30 #endif
     31 
     32 #else
     33 

optixrap/ORng.cc::

    152 void ORng::setSkipAhead( unsigned skipahead )
    153 {
    154     m_rng_skipahead = skipahead ;
    155 #ifdef WITH_SKIPAHEAD
    156     LOG(fatal) << "WITH_SKIPAHEAD skipahead " << skipahead ;
    157     m_context["rng_skipahead"]->setUint(m_rng_skipahead) ;
    158 #else
    159     LOG(LEVEL) << " skip as as WITH_SKIPAHEAD not enabled " ;
    160 #endif
    161 }
    162 unsigned ORng::getSkipAhead() const
    163 {
    164     return m_rng_skipahead ;
    165 }

optixrap/cu/generate.cu::

    581 
    582     curandState rng = rng_states[photon_id];
    583 
    584 #ifdef WITH_SKIPAHEAD
    585     unsigned long long rng_skipahead_ = rng_skipahead ;   // see ORng.hh
    586     if( rng_skipahead_ > 0ull )
    587     {
    588         skipahead(rng_skipahead_ , &rng) ;
    589     }
    590 #endif


::

    epsilon:optixrap blyth$ opticks-f setSkipAhead
    ./optickscore/OpticksEvent.hh:       void     setSkipAhead(unsigned skipahead);
    ./optickscore/OpticksEvent.cc:void OpticksEvent::setSkipAhead(unsigned skipahead)  // TODO: move to unsigned long long 
    ./optickscore/OpticksRun.cc:    evt->setSkipAhead( skipahead ); // TODO: make configurable + move to ULL
    ./optixrap/OPropagator.hh:        void     setSkipAhead(unsigned skipahead);
    ./optixrap/OPropagator.cc:    m_orng->setSkipAhead(skipahead); 
    ./optixrap/ORng.hh:      void setSkipAhead(unsigned skipahead); 
    ./optixrap/ORng.cc:    setSkipAhead(0); 
    ./optixrap/ORng.cc:ORng::setSkipAhead
    ./optixrap/ORng.cc:void ORng::setSkipAhead( unsigned skipahead )
    epsilon:opticks blyth$ 



The *tagoffset* is an event index 0,1,2,... and *skipaheadstep* is configurable.

To ensure that each 




::

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



optickscore/OpticksCfg.cc::

    1163    char skipaheadstep[128];
    1164    snprintf(skipaheadstep,128,
    1165 "Unsigned int skipaheadstep used with ORng"
    1166 "Default %d ", m_skipaheadstep);
    1167    m_desc.add_options()
    1168        ("skipaheadstep",  boost::program_options::value<unsigned>(&m_skipaheadstep), skipaheadstep );
    1169 
    1170 




Improvements for the qudarap version of this
-----------------------------------------------

* potential for needing very large values, so need to use 64bit ints 
* BUT pure CUDA easier in this regard tham OptiX so maybe do not 
  need to fiddle around with something like the below
    
* https://stackoverflow.com/questions/18312821/type-casting-to-unsigned-long-long-in-cuda


::

    
    __host__ __device__ unsigned long long int hiloint2uint64(int h, int l)
    {
        int combined[] = { h, l };

        return *reinterpret_cast<unsigned long long int*>(combined);
    }





