#pragma once

class Timer ; 
class Opticks ;
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
       OEngineImp(Opticks* opticks, GGeo* ggeo);
    private:
       void init();
    public:
       OContext*    getOContext();
       OPropagator* getOPropagator();
       void setEvent(OpticksEvent* evt);

       void prepareOptiX();             // creates OptiX context and populates with geometry info
       void preparePropagator();        // OPropagator : initEvent creates GPU buffers: genstep, photon, record, sequence
       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers
       void saveEvt();
       void cleanup();

    private:
       Timer*            m_timer ;
       Opticks*          m_opticks ; 
       OpticksCfg<Opticks>* m_fcfg ;   
       GGeo*             m_ggeo ; 
       OpticksEvent*     m_evt ; 
    

       OContext*         m_ocontext ; 
       OColors*          m_ocolors ; 
       OGeo*             m_ogeo ; 
       OBndLib*          m_olib ; 
       OScintillatorLib* m_oscin ; 
       OSourceLib*       m_osrc ; 
       OTracer*          m_otracer ; 
       OPropagator*      m_opropagator ; 

};

