#pragma once

// bcfg-
class Cfg ;
 
// npy-
template <typename> class NPY ;
class NumpyEvt ; 
class G4StepNPY ; 
class TorchStepNPY ; 
class BoundariesNPY ; 
class PhotonsNPY ; 
class RecordsNPY ; 
class HitsNPY ;
class Parameters ; 
class Timer ; 
class Types ; 
class Index ; 
class Lookup ; 

// numpyserver-
#ifdef NPYSERVER
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif

// oglrap-
template <typename> class FrameCfg ;
class Scene ; 
class Composition ; 
class Frame ; 
class Bookmarks ; 
class Interactor ; 
class GUI ; 
class Photons ; 

// glfw-
struct GLFWwindow ; 

// ggeo-
class GCache ;
class GGeo ;
class GLoader ; 
class GBoundaryLib ; 
class GBoundaryLibMetadata ;
class GMergedMesh ; 
class GItemIndex ; 

// optixrap
class OContext ; 
class OGeo ; 
class OBndLib ; 
class OScintillatorLib ; 
class OFrame ;
class ORenderer ; 
class OTracer ; 
class OPropagator ; 
class OColors ; 


#include <map>
#include <string>

// glm-
#include <glm/glm.hpp>

class App {
  public:
       App(const char* prefix, int argc, char** argv );
       bool isExit();
  private:
       void init(int argc, char** argv);
       void wiring();
       void setExit(bool exit=true);
  public:
       int config(int argc, char** argv);
  public:
       void prepareScene();   // creates OpenGL context window and OpenGL renderers loading shaders
  public:
       void loadGeometry();
       void registerGeometry();
       void checkGeometry();
       void uploadGeometry();
  public:
       void loadGenstep();
       NPY<float>* loadGenstepFromFile(const std::string& typ, const std::string& tag, const std::string& det);

       TorchStepNPY* makeSimpleTorchStep();
       TorchStepNPY* makeCalibrationTorchStep(unsigned int imesh);

       void uploadEvt();
       void seedPhotonsFromGensteps();
       void initRecords();
  public:
       void configureGeometry(); 
       void prepareOptiX();
       void preparePropagator();
       void propagate();
       void downloadEvt();
       void indexEvt();
       void indexSequence();
       void indexBoundaries();
       void indexEvtOld();
  public:
       void prepareGUI();
       void makeReport();
       void renderLoop();
       void render();
       void cleanup();

  public:
       GCache* getCache(); 
       FrameCfg<Frame>* getFrameCfg();
       bool hasOpt(const char* name);

  private:
       const char*  m_prefix ; 
       Parameters*  m_parameters ; 
       Timer*       m_timer ; 
       GCache*      m_cache ; 
       Scene*       m_scene ; 
       Composition* m_composition ;
       Frame*       m_frame ;
       GLFWwindow*  m_window ; 
       Bookmarks*   m_bookmarks ;
       Interactor*  m_interactor ;
#ifdef NPYSERVER
       numpydelegate* m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       NumpyEvt*        m_evt ;
       Cfg*             m_cfg ;
       FrameCfg<Frame>* m_fcfg ; 
       Types*           m_types ; 
       GGeo*            m_ggeo ; 
       GMergedMesh*     m_mesh0 ;  
       OContext*        m_ocontext ; 
       OColors*         m_ocolors ; 
       OGeo*            m_ogeo ; 
       OBndLib*          m_olib ; 
       OScintillatorLib* m_oscin ; 
       OFrame*          m_oframe ; 
       ORenderer*       m_orenderer ; 
       OTracer*         m_otracer ; 
       OPropagator*     m_opropagator ; 

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
       bool             m_exit ; 


   private:
       glm::uvec4       m_size ;

};




inline App::App(const char* prefix, int argc, char** argv )
   : 
      m_prefix(strdup(prefix)),
      m_parameters(NULL),
      m_timer(NULL),
      m_cache(NULL),
      m_scene(NULL),
      m_composition(NULL),
      m_frame(NULL),
      m_window(NULL),
      m_bookmarks(NULL),
      m_interactor(NULL),
#ifdef NPYSERVER
      m_delegate(NULL),
      m_server(NULL),
#endif
      m_evt(NULL), 
      m_cfg(NULL),
      m_fcfg(NULL),
      m_types(NULL),
      m_ggeo(NULL),
      m_mesh0(NULL),
      m_ocontext(NULL),
      m_ocolors(NULL),
      m_ogeo(NULL),
      m_olib(NULL),
      m_oscin(NULL),
      m_oframe(NULL),
      m_orenderer(NULL),
      m_otracer(NULL),
      m_opropagator(NULL),
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
      m_torchstep(NULL),
      m_exit(false)
{
    init(argc, argv);
}

inline GCache* App::getCache()
{
    return m_cache ; 
}

inline FrameCfg<Frame>* App::getFrameCfg()
{
    return m_fcfg ; 
}
inline bool App::isExit()
{
    return m_exit ; 
}
inline void App::setExit(bool exit)
{
    m_exit = exit  ; 
}

