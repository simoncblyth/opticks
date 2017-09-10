#pragma once

class SLog ; 

class Opticks ;       // okc-
class OpticksEntry ; 
class OpticksHub ;    // okg-

class OScene ;   // optixrap-
class OPropagator ; 
class OEvent ; 
class OContext ; 

class OpSeeder ; 
class OpZeroer ; 
class OpIndexer ; 


#include "OKOP_API_EXPORT.hh"

/**
OpEngine
=========

OpEngine takes a central role, it holds the OScene
which creates the OptiX context holding the GPU geometry
and all GPU buffers.


Canonical OpEngine instance m_engine resides in ok-/OKPropagator 
which resides as m_propagator at top level in ok-/OKMgr

* BUT: ok- depends on OpenGL ... need compute only equivalents okop-/OpPropagator okop/OpMgr


NB OpEngine is ONLY AT COMPUTE LEVEL, FOR THE FULL PICTURE NEED TO SEE ONE LEVEL UP 
   IN ok-
   OKPropagator::uploadEvent 
   OKPropagator::downloadEvent



  
**/

class OKOP_API OpEngine {
       // friends can access the OPropagator
       friend class OpIndexer ; 
       friend class OpSeeder ; 
       friend class OpZeroer ; 
    public:
       OpEngine(OpticksHub* hub);
    public:
       OContext*    getOContext();         // needed by opticksgl-/OpViz

       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers
       void indexEvent();
       unsigned downloadEvent();
       unsigned uploadEvent();
       unsigned getOptiXVersion();
       void cleanup();
       void Summary(const char* msg="OpEngine::Summary");

    private:
       OPropagator* getOPropagator();
    private:
       void downloadPhotonData();       // see App::dbgSeed
       void init();
       void initPropagation();
    private:
       SLog*                m_log ; 
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       OScene*              m_scene ; 
       OContext*            m_ocontext ; 
       OpticksEntry*        m_entry ; 

       OEvent*              m_oevt ; 
       OPropagator*         m_propagator ; 
       OpSeeder*            m_seeder ; 
       OpZeroer*            m_zeroer ; 
       OpIndexer*           m_indexer ; 
       bool                 m_immediate ; 
};


