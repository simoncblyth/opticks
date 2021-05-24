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

// sysrap-
class SLog ; 
class SRenderer ; 
class SLauncher ; 

class BCfg ; 

// npy-
template <typename T> class NPY ; 

// ggeo-
class GGeo ; 
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

#ifdef WITH_BOOST_ASIO
#include <boost/asio.hpp>
template <typename T> class BListenUDP ;
#endif



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
         void uploadEvent(char ctrl);
         void indexPresentationPrep();
         void cleanup();
    private:
         void prepareScene(const char* rendermode=NULL);
         void setupRendermode(const char* rendermode );
         void setupRestrictions();
    private: 
         void uploadEvent(OpticksEvent* evt);
    private: 
         int preinit() const ;
         void init();
         void render();
         void renderGUI();
         void prepareGUI();
         void renderLoop();
    public:
         void downloadData(NPY<float>* data);
         void downloadEvent();
    private:
         int           m_preinit ; 
         SLog*         m_log ; 
#ifdef WITH_BOOST_ASIO
         boost::asio::io_context m_io_context ;    
         BListenUDP<OpticksViz>*  m_listen_udp ; 
#endif
         OpticksHub*   m_hub ; 
         BCfg*         m_umbrella_cfg ; 
         Opticks*      m_ok ; 
         OpticksRun*   m_run ; 
         GGeoBase*     m_ggb ; 
         GGeo*         m_ggeo ; 

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


#ifdef WITH_BOOST_ASIO
#include "BListenUDP.hh"
template class BListenUDP<OpticksViz>;
#endif



