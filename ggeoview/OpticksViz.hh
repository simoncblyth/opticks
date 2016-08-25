#pragma once

class Opticks ; 
class OpticksHub ; 
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


#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"

class GGV_API OpticksViz {
    public:
         OpticksViz(OpticksHub* hub);
         void setExternalRenderer(SRenderer* external_renderer);
         void setEvent(OpticksEvent* evt);
         OpticksEvent* getEvent();
    public:
         bool hasOpt(const char* name);
         void configure();
         void prepareScene();
         void uploadGeometry();
         void targetGenstep();
         void uploadEvent();
         void indexPresentationPrep();
         void prepareGUI();
         void renderLoop();
         void cleanup();
    private: 
         void render();
         void renderGUI();
    public:
         Scene* getScene(); 
         void downloadData(NPY<float>* data);
         void downloadEvent();
    private:
         void init();
    private:
         OpticksHub*   m_hub ; 
         Opticks*      m_opticks ; 
         int           m_interactivity ; 
         Composition*  m_composition ;
         Types*        m_types ; 

         Scene*       m_scene ; 
         Frame*       m_frame ;
         GLFWwindow*  m_window ; 
         Interactor*  m_interactor ;

         GItemIndex*  m_seqhis ; 
         GItemIndex*  m_seqmat ; 
         GItemIndex*  m_boundaries ; 

         Photons*     m_photons ; 
         GUI*         m_gui ; 

         SRenderer*   m_external_renderer ; 

};

#include "GGV_TAIL.hh"



