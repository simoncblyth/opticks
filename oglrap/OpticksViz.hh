#pragma once

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksGeometry ; 
class OpticksEvent ; 
class OpticksIdx ; 
class Composition ; 
class OpticksEvent ; 
class Types ; 
class Scene ; 
class Frame ; 
struct GLFWwindow ; 
class Interactor ; 
class GItemIndex ; 
class Photons ; 
class GUI ; 
template <typename T> class NPY ; 

class SRenderer ; 
class SLauncher ; 


#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API OpticksViz {
         friend class AxisApp ; 
    public:
         OpticksViz(OpticksHub* hub, OpticksIdx* idx, bool immediate=false);
    public:
         void visualize();
    public:
         void setExternalRenderer(SRenderer* external_renderer);
         void setLauncher(SLauncher* launcher);
         void setTitle(const char* title);
    public:
         bool hasOpt(const char* name);
         Opticks*       getOpticks(); 
         Interactor*    getInteractor();
         OpticksHub*    getHub(); 
         NConfigurable* getSceneConfigurable(); 
         Scene*         getScene(); 
         int            getTarget();
    public:
         void prepareScene(const char* rendermode=NULL);
         void uploadGeometry();
         void uploadEvent();
         void indexPresentationPrep();
         void cleanup();
    private: 
         void uploadEvent(OpticksEvent* evt);
    private: 
         void init();
         void render();
         void renderGUI();
         void prepareGUI();
         void renderLoop();
    public:
         void downloadData(NPY<float>* data);
         void downloadEvent();
    private:
         SLog*         m_log ; 
         OpticksHub*   m_hub ; 
         OpticksGeometry* m_geometry ; 
         OpticksIdx*   m_idx ; 
         bool          m_immediate ; 
         Opticks*      m_opticks ; 
         int           m_interactivity ; 
         Composition*  m_composition ;
         Types*        m_types ; 

         const char*   m_title ; 
         Scene*       m_scene ; 
         Frame*       m_frame ;
         GLFWwindow*  m_window ; 
         Interactor*  m_interactor ;

         GItemIndex*  m_seqhis ; 
         GItemIndex*  m_seqmat ; 
         GItemIndex*  m_boundaries ; 

         Photons*     m_photons ; 
         GUI*         m_gui ; 

         SLauncher*   m_launcher ; 
         SRenderer*   m_external_renderer ; 

};




