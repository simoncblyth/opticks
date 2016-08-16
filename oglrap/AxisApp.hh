#pragma once

class SLauncher ; 
class Opticks ; 
class Composition ; 
class Scene ; 
class Frame ; 
class Interactor ; 
struct GLFWwindow ; 
class Rdr ; 
class MultiViewNPY ; 
template<typename T> class NPY ; 

#include "OGLRAP_API_EXPORT.hh"

/**
AxisApp
~~~~~~~~~

Aims to provide the simplest possible use of OpenGL VBOs 
within Opticks machinery, to help with debugging of 
OpenGL-OptiX interop issues.

**/

class OGLRAP_API AxisApp {
  public:
      AxisApp(int argc, char** argv);
  public:
      void renderLoop();
      MultiViewNPY* getAxisAttr();
      NPY<float>*   getAxisData();
      void setLauncher(SLauncher* launcher);  
     // SLauncher is a pure virtual protocol:  void launch(unsigned count)
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
      NPY<float>*   m_axis_data ; 
      SLauncher*    m_launcher ; 

};



