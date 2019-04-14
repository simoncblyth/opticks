#pragma once

// sysrap-
class SLog ; 
class SRenderer ; 
class SLauncher ; 

// npy-
template <typename T> class NPY ; 

// ggeo-
class GGeoBase ; 
class GItemIndex ; 

// okc-
class Opticks ; 
class OpticksRun ; 
class OpticksEvent ; 
class Composition ; 
class Bookmarks ; 

class ContentStyle ; 
class GlobalStyle ; 

class OpticksEvent ; 
class Types ; 

// okg-
class OpticksHub ; 
class OpticksIdx ; 

// oglrap-
class Scene ; 
class Frame ; 

struct GLFWwindow ; 
class Interactor ; 
class Photons ; 
class GUI ; 

#include "plog/Severity.h"
#include "OGLRAP_API_EXPORT.hh"
#include "SCtrl.hh"


/**
OpticksViz
===========

Canonical m_viz instances are residents of the top level managers: ok/OKMgr.hh okg4/OKG4Mgr.hh opticksgl/OKGLTracer.hh

**/


class OGLRAP_API OpticksViz : public SCtrl  {
         friend class AxisApp ; 
    public:
         static const plog::Severity LEVEL ; 
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
         // SCtrl 
         void command(const char* ctrl);         // single 2-char command 
         void commandline(const char* cmdline);  // list of comma delimited 2-char commands
    public:
         void uploadGeometry();
         void uploadEvent();
         void indexPresentationPrep();
         void cleanup();
    private:
         void prepareScene(const char* rendermode=NULL);
         void setupRendermode(const char* rendermode );
         void setupRestrictions();
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
         Opticks*      m_ok ; 
         OpticksRun*   m_run ; 
         GGeoBase*     m_ggb ; 

         OpticksIdx*   m_idx ; 
         bool          m_immediate ; 
         int           m_interactivity ; 
         Composition*  m_composition ;
         Bookmarks*    m_bookmarks ;

         ContentStyle* m_content_style ; 
         GlobalStyle*  m_global_style ; 

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




