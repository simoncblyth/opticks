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

Canonical m_engine instance resides in ggeoview-/OKPropagator

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
       void downloadEvent();
       void uploadEvent();

       void cleanup();
       void Summary(const char* msg="OpEngine::Summary");

    private:
       OPropagator* getOPropagator();
    private:
       void downloadPhotonData();       // see App::dbgSeed

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


