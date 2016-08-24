#pragma once

#include <map>
#include <string>
#include <glm/fwd.hpp>

// brap-
class BCfg ;
 
// npy-
template <typename> class NPY ;
class G4StepNPY ; 
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

// glfw-
struct GLFWwindow ; 

// ggeo-
class GGeo ;
class GItemIndex ; 


// opticksgeo-
class OpticksGeometry ; 
class OpticksHub ; 


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
       void dbgSeed();
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
       bool hasOpt(const char* name);

  private:
       Opticks*          m_opticks ; 
       OpticksHub*       m_hub ; 
       const char*       m_prefix ; 
       Parameters*       m_parameters ; 
       Timer*            m_timer ; 
       Scene*            m_scene ; 
       Composition*      m_composition ;
       Frame*            m_frame ;
       GLFWwindow*       m_window ; 
       Interactor*       m_interactor ;
       OpticksEvent*    m_evt ;
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
   private:
       glm::uvec4       m_size ;

};

#include "GGV_TAIL.hh"


