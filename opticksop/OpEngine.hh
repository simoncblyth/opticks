#pragma once

class SLog ; 

class Opticks ;       // okc-
class OpticksHub ;    // okg-

class OEngineImp ;   // optixrap-
class OPropagator ; 
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
    public:
       OpEngine(OpticksHub* hub);
    public:
       OContext* getOContext();         // needed by opticksgl-/OpViz
 
       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers

       void downloadEvent();
       void uploadEvent();

       void cleanup();
       void Summary(const char* msg="OpEngine::Summary");

    private:
       void downloadPhotonData();       // see App::dbgSeed

    private:
       SLog*                m_log ; 
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       OEngineImp*          m_imp ; 
       OPropagator*         m_propagator ; 
       OpSeeder*            m_seeder ; 
       OpZeroer*            m_zeroer ; 
       OpIndexer*           m_indexer ; 
       bool                 m_immediate ; 
};


