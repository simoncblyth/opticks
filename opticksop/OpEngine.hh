#pragma once




// opticks-
class Opticks ; 
template <typename> class OpticksCfg ;

// ggeo-
class GGeo ; 
class GCache ; 


// optixrap-
class OContext ; 
class OGeo ; 
class OBndLib ; 
class OScintillatorLib ; 
class OSourceLib ; 
class OFrame ;
class ORenderer ; 
class OTracer ; 
class OPropagator ; 
class OColors ; 

// oglrap-
class Scene ; 
class Composition ; 
class Interactor ; 

// npy-
class Timer ; 
class NumpyEvt ; 


class OpEngine {
    public:
       OpEngine(Opticks* opticks, GGeo* ggeo);
    private:
       void init();
       void postSetScene();
    public:
       void prepareOptiX();
       void setScene(Scene* scene);
       void prepareOptiXViz();
       void render();
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
       Scene*               m_scene ; 
       Composition*         m_composition ; 
       Interactor*          m_interactor ; 
    private:
       NumpyEvt*            m_evt ; 
    private:
       OContext*        m_ocontext ; 
       OColors*         m_ocolors ; 
       OGeo*            m_ogeo ; 
       OBndLib*          m_olib ; 
       OScintillatorLib* m_oscin ; 
       OSourceLib*       m_osrc ; 
       OFrame*          m_oframe ; 
       ORenderer*       m_orenderer ; 
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
      m_scene(NULL),
      m_evt(NULL),

      m_ocontext(NULL),
      m_ocolors(NULL),
      m_ogeo(NULL),
      m_olib(NULL),
      m_oscin(NULL),
      m_osrc(NULL),
      m_oframe(NULL),
      m_orenderer(NULL),
      m_otracer(NULL),
      m_opropagator(NULL)
{
      init();
}


inline void OpEngine::setScene(Scene* scene)
{
    m_scene = scene ; 
    postSetScene();
}

inline void OpEngine::setEvent(NumpyEvt* evt)
{
    m_evt = evt ; 
}

