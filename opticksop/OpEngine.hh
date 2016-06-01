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
       GCache*              m_cache ; 
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



inline void OpEngine::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
}

