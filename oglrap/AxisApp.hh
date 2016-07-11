#pragma once

class Opticks ; 
class Composition ; 
class Scene ; 
class Frame ; 
class Interactor ; 
struct GLFWwindow ; 
class Rdr ; 
class MultiViewNPY ; 

#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API AxisApp {
  public:
      AxisApp(int argc, char** argv);
      void renderLoop();
      MultiViewNPY* getAxisAttr();
  private:
      void init(int argc, char** argv); 
      void initViz();
      void prepareViz();
      void upload();
      void render(); 
  private:
      Opticks*     m_opticks ;
      Composition* m_composition ;
      Scene*       m_scene ;
      Frame*       m_frame ;
      Interactor*  m_interactor ;
      GLFWwindow*  m_window ;
      Rdr*         m_axis_renderer ; 
      MultiViewNPY* m_axis_attr ; 

};



