#pragma once

class SLog ; 
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
    public:
       void uploadEvent();
       void propagate(); 
       void downloadEvent();
    public:
       void downloadPhotonData();
       void cleanup();
    private:
       void prepareOptiXGeometry();     // creates OptiX context and populates with geometry info
       void preparePropagator();        // create OPropagator does prelaunch

    private:
       SLog*                m_log ; 
       Timer*               m_timer ;
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ;   
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

