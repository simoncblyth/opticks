#pragma once

// npy-
class Timer ; 

// opticks-
class Opticks ; 
class OpticksEvent ; 
template <typename> class OpticksCfg ;
class Composition ; 

// ggeo-
class GGeo ; 

// optixrap-
class OEngineImp ; 
class OContext ; 


#include "OKOP_API_EXPORT.hh"
class OKOP_API OpEngine {
    public:
       OpEngine(Opticks* opticks, GGeo* ggeo);
    private:
       void init();
    public:
       Opticks* getOpticks();
       OContext* getOContext();  // needed by opticksgl-/OpViz

       void prepareOptiX();             // creates OptiX context and populates with geometry info

       void setEvent(OpticksEvent* evt);
       OpticksEvent* getEvent();
 
       void preparePropagator();        // OPropagator : initEvent creates GPU buffers: genstep, photon, record, sequence
       void seedPhotonsFromGensteps();  // OpSeeder : seeds GPU photon buffer with genstep indices
       void initRecords();              // OpZeroer : zeros GPU record buffer via OptiX or OpenGL
       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers
       void saveEvt();
       void downloadPhotonData();       // see App::dbgSeed

       void indexSequence();
       void cleanup();

    private:
       Timer*               m_timer ; 
       Opticks*             m_opticks ; 
       OpticksCfg<Opticks>* m_fcfg ;   
    private:
       GGeo*                m_ggeo ; 
    private:
       Composition*         m_composition ; 
    private:
       OpticksEvent*        m_evt ; 
    private:
       OEngineImp*          m_imp ; 
};


