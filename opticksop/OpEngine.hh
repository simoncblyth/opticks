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
class OContext ; 
class OGeo ; 
class OBndLib ; 
class OScintillatorLib ; 
class OSourceLib ; 
class OTracer ; 
class OPropagator ; 
class OColors ; 

#include "OKOP_API_EXPORT.hh"
class OKOP_API OpEngine {
    public:
       OpEngine(Opticks* opticks, GGeo* ggeo);
    private:
       void init();
    public:
       Opticks* getOpticks();
       OContext* getOContext();

       void prepareOptiX();             // creates OptiX context and populates with geometry info

       void setEvent(OpticksEvent* evt);
       void preparePropagator();        // OPropagator : initEvent creates GPU buffers: genstep, photon, record, sequence
       void seedPhotonsFromGensteps();  // OpSeeder : seeds GPU photon buffer with genstep indices
       void initRecords();              // OpZeroer : zeros GPU record buffer via OptiX or OpenGL
       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers
       void saveEvt();

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
       OContext*        m_ocontext ; 
       OColors*         m_ocolors ; 
       OGeo*            m_ogeo ; 
       OBndLib*          m_olib ; 
       OScintillatorLib* m_oscin ; 
       OSourceLib*       m_osrc ; 
       OTracer*         m_otracer ; 
       OPropagator*     m_opropagator ; 

};


