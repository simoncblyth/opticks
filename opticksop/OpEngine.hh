#pragma once

// okc-
class OpticksEvent ; 
class Composition ; 

// okg-
class OpticksHub ; 

// optixrap-
class OEngineImp ; 
class OContext ; 


#include "OKOP_API_EXPORT.hh"
class OKOP_API OpEngine {
    public:
       OpEngine(OpticksHub* hub);
    public:
       OContext* getOContext();         // needed by opticksgl-/OpViz

       void prepareOptiX();             // creates OptiX context and populates with geometry info
 
       void preparePropagator();        // OPropagator : initEvent creates GPU buffers: genstep, photon, record, sequence
       void seedPhotonsFromGensteps();  // OpSeeder : seeds GPU photon buffer with genstep indices
       void initRecords();              // OpZeroer : zeros GPU record buffer via OptiX or OpenGL
       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers

       void downloadEvt();
       void downloadPhotonData();       // see App::dbgSeed

       void indexSequence();
       void cleanup();

    private:
       OpticksHub*          m_hub ; 
       OEngineImp*          m_imp ; 
};


