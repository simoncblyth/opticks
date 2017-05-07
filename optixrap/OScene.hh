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

/**

OScene
========

Canonical m_scene instance resides in okop-/OpEngine 



**/



#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OScene {
    public:
       OScene(OpticksHub* hub);
    public:
       OContext*    getOContext();
       OBndLib*     getOBndLib();
    public:
       void cleanup();
    private:
       void init();   // creates OptiX context and populates with geometry info
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

};

