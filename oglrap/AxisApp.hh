#pragma once

class SLauncher ; 
class Opticks ; 
class OpticksHub ; 
class OpticksViz ; 

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

TODO: enable OpticksViz to be used in such a 
      setting to make this app much simpler
      and avoid duplicated "wiring"

**/

class OGLRAP_API AxisApp {
  public:
      AxisApp(int argc, char** argv);
  public:
      void renderLoop();
      MultiViewNPY* getAxisAttr();
      NPY<float>*   getAxisData();
      void setLauncher(SLauncher* launcher);   // SLauncher instances provide: void launch(unsigned count)
  private:
      void init(); 
      void prepareViz();
      void upload();
  private:
      Opticks*     m_opticks ;
      OpticksHub*  m_hub ;
      OpticksViz*  m_viz ;
      Composition* m_composition ;
      Scene*       m_scene ;

      Rdr*         m_axis_renderer ; 
      MultiViewNPY* m_axis_attr ; 
      NPY<float>*   m_axis_data ; 

};



