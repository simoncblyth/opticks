#pragma once

class Opticks ;       // okc-
class OpticksEvent ;  

class OpticksHub ;    // okg-

class OEngineImp ;   // optixrap-
class OContext ; 

class OpSeeder ; 
class OpZeroer ; 
class OpIndexer ; 


#include "OKOP_API_EXPORT.hh"
class OKOP_API OpEngine {
    public:
       OpEngine(OpticksHub* hub);
    public:
       OContext* getOContext();         // needed by opticksgl-/OpViz
 
       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers

       void downloadEvt();

       void cleanup();
       void Summary(const char* msg="OpEngine::Summary");

    private:
       void init();
       void downloadPhotonData();       // see App::dbgSeed

    private:
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       OEngineImp*          m_imp ; 
       OpSeeder*            m_seeder ; 
       OpZeroer*            m_zeroer ; 
       OpIndexer*           m_indexer ; 
       bool                 m_immediate ; 
};


