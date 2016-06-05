#pragma once


// bcfg-
class Cfg ;
 
// npy-
template <typename> class NPY ;
class G4StepNPY ; 
class TorchStepNPY ; 
class BoundariesNPY ; 
class PhotonsNPY ; 
class RecordsNPY ; 
class HitsNPY ;
class Parameters ; 
class Timer ; 
class Types ; 
class NState ; 

//opticks-
class Opticks ; 
class OpticksEvent ; 
template <typename> class OpticksCfg ;
class Composition ; 
class Bookmarks ; 

// oglrap-
class Scene ; 
class Frame ; 
class Interactor ; 
class GUI ; 
class Photons ; 
class DynamicDefine ; 

// glfw-
struct GLFWwindow ; 

// ggeo-
class GGeo ;
class GCache ;

class OpticksGeometry ; 

class GItemIndex ; 


#ifdef WITH_NPYSERVER
// numpyserver-
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif


#ifdef WITH_OPTIX
// opop-
class OpEngine ; 
// optixgl-
class OpViz ; 
#endif

#include <map>
#include <string>
#include <cstring>

// glm-
#include <glm/glm.hpp>


class App {
  public:
       App(const char* prefix, int argc, char** argv );
       void initViz();
       void configure(int argc, char** argv);
  private:
       void init(int argc, char** argv);
  public:
       void prepareViz();   // creates OpenGL context window and OpenGL renderers loading shaders
       void loadGeometry();
       bool isExit();
  private: 
       void configureViz(); 

  public:
       void uploadGeometryViz();
  public:
       void loadGenstep();
       void targetViz();

       void loadEvtFromFile();


       void uploadEvtViz();


#ifdef WITH_OPTIX
  public:
       void prepareOptiX();
       void prepareOptiXViz();

       void setupEventInEngine();
       void preparePropagator();
       void seedPhotonsFromGensteps();
       void initRecords();
       void propagate();
       void saveEvt();
       void indexSequence();
#endif

       void indexEvt();
       void indexPresentationPrep();
       void indexBoundariesHost();
       void indexEvtOld();
  public:
       void prepareGUI();
       void renderLoop();
       void renderGUI();
       void render();
       void cleanup();

  public:
       GCache* getCache(); 
       OpticksCfg<Opticks>* getOpticksCfg();
       bool hasOpt(const char* name);

  private:
       Opticks*          m_opticks ; 
       const char*       m_prefix ; 
       Parameters*       m_parameters ; 
       Timer*            m_timer ; 
       GCache*           m_cache ; 
       DynamicDefine*    m_dd ; 
       NState*           m_state ; 
       Scene*            m_scene ; 
       Composition*      m_composition ;
       Frame*            m_frame ;
       GLFWwindow*       m_window ; 
       Bookmarks*        m_bookmarks ;
       Interactor*       m_interactor ;
#ifdef WITH_NPYSERVER
       numpydelegate* m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       OpticksEvent*    m_evt ;
       Cfg*             m_cfg ;

       OpticksCfg<Opticks>* m_fcfg ;   
       Types*           m_types ; 

       OpticksGeometry* m_geometry ; 
       GGeo*            m_ggeo ; 

#ifdef WITH_OPTIX
       OpEngine*        m_ope ; 
       OpViz*           m_opv ; 
#endif

       BoundariesNPY*   m_bnd ; 
       PhotonsNPY*      m_pho ; 
       HitsNPY*         m_hit ; 
       RecordsNPY*      m_rec ; 

       GItemIndex*      m_seqhis ; 
       GItemIndex*      m_seqmat ; 
       GItemIndex*      m_boundaries ; 

       Photons*         m_photons ; 
       GUI*             m_gui ; 
       G4StepNPY*       m_g4step ; 
       TorchStepNPY*    m_torchstep ; 


   private:
       glm::uvec4       m_size ;

};




inline App::App(const char* prefix, int argc, char** argv )
   : 
      m_opticks(NULL),
      m_prefix(strdup(prefix)),
      m_parameters(NULL),
      m_timer(NULL),
      m_cache(NULL),
      m_dd(NULL),
      m_state(NULL),
      m_scene(NULL),
      m_composition(NULL),
      m_frame(NULL),
      m_window(NULL),
      m_bookmarks(NULL),
      m_interactor(NULL),
#ifdef WITH_NPYSERVER
      m_delegate(NULL),
      m_server(NULL),
#endif
      m_evt(NULL), 
      m_cfg(NULL),
      m_fcfg(NULL),
      m_types(NULL),
      m_geometry(NULL),
      m_ggeo(NULL),

#ifdef WITH_OPTIX
      m_ope(NULL),
      m_opv(NULL),
#endif

      m_bnd(NULL),
      m_pho(NULL),
      m_hit(NULL),
      m_rec(NULL),
      m_seqhis(NULL),
      m_seqmat(NULL),
      m_boundaries(NULL),
      m_photons(NULL),
      m_gui(NULL),
      m_g4step(NULL),
      m_torchstep(NULL)
{
    init(argc, argv);
}

inline GCache* App::getCache()
{
    return m_cache ; 
}

inline OpticksCfg<Opticks>* App::getOpticksCfg()
{
    return m_fcfg ; 
}


