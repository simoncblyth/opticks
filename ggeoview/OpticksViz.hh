#pragma once

class Opticks ; 
class OpticksHub ; 
class Composition ; 
class Scene ; 
class Frame ; 
struct GLFWwindow ; 
class Interactor ; 
class GItemIndex ; 
class Photons ; 
class GUI ; 

class OpticksViz {
    public:
         OpticksViz(OpticksHub* hub);
    public:
         void prepareScene();
         void uploadGeometry();
         void targetGenstep();
         void uploadEvent();
    private:
         void init();
    private:
         Opticks*     m_opticks ; 
         OpticksHub*  m_hub ; 
         Composition* m_composition ;

         Scene*       m_scene ; 
         Frame*       m_frame ;
         GLFWwindow*  m_window ; 
         Interactor*  m_interactor ;

         GItemIndex*  m_seqhis ; 
         GItemIndex*  m_seqmat ; 
         GItemIndex*  m_boundaries ; 

         Photons*     m_photons ; 
         GUI*         m_gui ; 

};
