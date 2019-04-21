#pragma once

class SLog ; 
class Timer ; 
class Opticks ;
class OpticksHub ;
template <typename> class OpticksCfg ;

class OContext ; 
class OFunc ; 
class OColors ; 
class OGeo ; 
class OBndLib ; 
class OScintillatorLib ; 
class OSourceLib ; 

/**
OScene
========

Canonical m_scene instance resides in okop-/OpEngine 

Instanciating an OScene creates the OptiX GPU context 
and populates it with geometry, boundary info etc.. 
effectively uploading the geometry obtained from
the OpticksHub to the GPU.  This geometry info is 
held in the O* libs: OGeo, OBndLib, OScintillatorLib, 
OSourceLib.

NB there is no use of OptiX types in this interface header
although these are used internally. This is as are aiming 
to remove OptiX dependency in higher level interfaces 
for easier OptiX version hopping.

**/

#include "plog/Severity.h"
#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OScene {
    public:
       static const plog::Severity LEVEL ; 
    public:
       OScene(OpticksHub* hub);
    public:
       OContext*    getOContext();
       OBndLib*     getOBndLib();
    public:
       void cleanup();
    private:
       void init();   // creates OptiX context and populates with geometry info
       void initRTX();
    private:
       SLog*                m_log ; 
       Timer*               m_timer ;
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ;   

       OContext*         m_ocontext ; 
       OFunc*            m_osolve ; 
       OColors*          m_ocolors ; 
       OGeo*             m_ogeo ; 
       OBndLib*          m_olib ; 
       OScintillatorLib* m_oscin ; 
       OSourceLib*       m_osrc ; 
       unsigned          m_verbosity ; 
       bool              m_use_osolve ; 

};

