#pragma once

class Timer ; 
class Opticks ;
class OpticksHub ;
template <typename> class OpticksCfg ;
class GGeo ; 

class OContext ; 
class OColors ; 
class OGeo ; 
class OBndLib ; 
class OScintillatorLib ; 
class OSourceLib ; 
class OTracer ; 
class OPropagator ; 

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OEngineImp {
    public:
       OEngineImp(OpticksHub* hub);
    private:
       void init();
    public:
       OContext*    getOContext();
       OPropagator* getOPropagator();

       void prepareOptiX();             // creates OptiX context and populates with geometry info
       void preparePropagator();        // OPropagator : initEvent creates GPU buffers: genstep, photon, record, sequence
       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers

       void downloadEvt();
       void downloadPhotonData();
       void cleanup();

    private:
       Timer*               m_timer ;
       OpticksHub*          m_hub ; 
       Opticks*             m_opticks ; 
       OpticksCfg<Opticks>* m_fcfg ;   
       GGeo*                m_ggeo ; 

       OContext*         m_ocontext ; 
       OColors*          m_ocolors ; 
       OGeo*             m_ogeo ; 
       OBndLib*          m_olib ; 
       OScintillatorLib* m_oscin ; 
       OSourceLib*       m_osrc ; 
       OTracer*          m_otracer ; 
       OPropagator*      m_opropagator ; 

};

