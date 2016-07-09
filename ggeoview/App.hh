#pragma once

#include <map>
#include <string>
#include <glm/fwd.hpp>

// brap-
class BCfg ;
 
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
class GItemIndex ; 

class OpticksGeometry ; 


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

#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"

class GGV_API App {
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
       bool isCompute();
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
       BCfg*             m_cfg ;

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

#include "GGV_TAIL.hh"


