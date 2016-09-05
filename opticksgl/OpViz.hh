#pragma once

// okc-
class Composition ; 

// okg-
class OpticksHub ;

// optixrap-
class OContext ;
class OTracer ;

//opop-
class OpEngine ; 

// optixgl-
class OFrame ;
class ORenderer ;

// oglrap-
class Scene ; 
class Interactor ; 
class OpticksViz ; 


#include "OKGL_API_EXPORT.hh"

#include "SRenderer.hh"

class OKGL_API OpViz : public SRenderer {
    public:
       OpViz(OpEngine* ope, OpticksViz* viz, bool immediate);
    public:
       void prepareTracer();
       void render();     // fulfils SRenderer protocol
    private:
       void init();
    private:
       OpEngine*        m_ope ; 
       OpticksViz*      m_viz ; 
       OpticksHub*      m_hub ; 
       bool             m_immediate ; 
       Scene*           m_scene ;

       OContext*        m_ocontext ; 
       Composition*     m_composition ; 
       Interactor*      m_interactor ;
       OFrame*          m_oframe ;
       ORenderer*       m_orenderer ;
       OTracer*         m_otracer ;

};


