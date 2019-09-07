/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
      AxisApp(Opticks* ok);
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
      Opticks*     m_ok ;
      OpticksHub*  m_hub ;
      OpticksViz*  m_viz ;
      Composition* m_composition ;
      Scene*       m_scene ;

      Rdr*         m_axis_renderer ; 
      MultiViewNPY* m_axis_attr ; 
      NPY<float>*   m_axis_data ; 

};



