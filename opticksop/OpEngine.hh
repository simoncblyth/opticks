#pragma once

// npy-
class Timer ; 
class NumpyEvt ; 

// opticks-
class Opticks ; 
template <typename> class OpticksCfg ;
class Composition ; 

// ggeo-
class GGeo ; 
class GCache ; 

// optixrap-
class OContext ; 
class OGeo ; 
class OBndLib ; 
class OScintillatorLib ; 
class OSourceLib ; 
class OTracer ; 
class OPropagator ; 
class OColors ; 


class OpEngine {
    public:
       OpEngine(Opticks* opticks, GGeo* ggeo);
    private:
       void init();
    public:
       Opticks* getOpticks();
       OContext* getOContext();

       void prepareOptiX();
       void setEvent(NumpyEvt* evt);
       void preparePropagator();
       void seedPhotonsFromGensteps();
       void initRecords();
       void propagate();
       void saveEvt();
       void indexSequence();
       void cleanup();

    private:
       Timer*               m_timer ; 
       Opticks*             m_opticks ; 
       OpticksCfg<Opticks>* m_fcfg ;   
    private:
       GGeo*                m_ggeo ; 
       GCache*              m_cache ; 
    private:
       Composition*         m_composition ; 
    private:
       NumpyEvt*            m_evt ; 
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


inline OpEngine::OpEngine(Opticks* opticks, GGeo* ggeo) 
     : 
      m_timer(NULL),
      m_opticks(opticks),
      m_fcfg(NULL),
      m_ggeo(ggeo),
      m_cache(NULL),
      m_evt(NULL),
      m_ocontext(NULL),
      m_ocolors(NULL),
      m_ogeo(NULL),
      m_olib(NULL),
      m_oscin(NULL),
      m_osrc(NULL),
      m_otracer(NULL),
      m_opropagator(NULL)
{
      init();
}


inline Opticks* OpEngine::getOpticks()
{
    return m_opticks ; 
}
inline OContext* OpEngine::getOContext()
{
    return m_ocontext ; 
}



inline void OpEngine::setEvent(NumpyEvt* evt)
{
    m_evt = evt ; 
}

