#pragma once

//opticks-
class Opticks ; 
class Composition ; 

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


class OpViz {
    public:
       OpViz(OpEngine* ope, Scene* scene);
    private:
       void init();
       void prepareOptiXViz();
    public:
       void render();

    private:
       OpEngine*        m_ope ; 
       Scene*           m_scene ;

    private:
       Opticks*         m_opticks ; 
       OContext*        m_ocontext ; 
       Composition*     m_composition ; 
       Interactor*      m_interactor ;
       OFrame*          m_oframe ;
       ORenderer*       m_orenderer ;
       OTracer*         m_otracer ;

};


inline OpViz::OpViz(OpEngine* ope, Scene* scene) 
   :
      m_ope(ope),
      m_scene(scene),

      m_opticks(NULL),
      m_ocontext(NULL),
      m_composition(NULL),
      m_interactor(NULL),
      m_oframe(NULL),
      m_orenderer(NULL),
      m_otracer(NULL)
{
   init();
}


