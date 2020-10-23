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

#include <cstring>

// sysrap-
#include "SLog.hh"
#include "SLauncher.hh"
#include "SRenderer.hh"

#include "BStr.hh"

#include "NMeta.hpp"


// npy-
#include "Types.hpp"

#ifdef OLD_TIMER
#include "BTimesTable.hh"
#else
#include "OpticksProfile.hh"
#endif


#include "NGLM.hpp"
#include "NPY.hpp"


#include "GItemIndex.hh"
#include "GGeoBase.hh"


// okc-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksRun.hh"
#include "Composition.hh"

#include "ContentStyle.hh"
#include "GlobalStyle.hh"

#include "Bookmarks.hh"

// opticksgeo-
#include "OpticksHub.hh"
#include "OpticksIdx.hh"

// ggeoview-
#include "Photons.hh"
#include "OpticksViz.hh"


// windows headers from PLOG need to be before glfw 
// http://stackoverflow.com/questions/3927810/how-to-prevent-macro-redefinition
#include "PLOG.hh"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
#include "Rdr.hh"

#define GUI_ 1
#ifdef GUI_
#include "GUI.hh"
#endif

#include "StateGUI.hh"

#include "Scene.hh"
#include "SceneCfg.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"
#include "Interactor.hh"
#include "InteractorCfg.hh"

const plog::Severity OpticksViz::LEVEL = PLOG::EnvLevel("OpticksViz", "debug") ; 


int OpticksViz::preinit() const 
{
    OKI_PROFILE("_OpticksViz::OpticksViz"); 
    return 0 ; 
}


OpticksViz::OpticksViz(OpticksHub* hub, OpticksIdx* idx, bool immediate)
    :
    SCtrl(),
    m_preinit(preinit()), 
    m_log(new SLog("OpticksViz::OpticksViz", "", LEVEL)),
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_run(m_ok->getRun()),
    m_ggb(m_hub->getGGeoBase()),
    m_ggeo(m_hub->getGGeo()),
    m_idx(idx),
    m_immediate(immediate),
    m_interactivity(m_ok->getInteractivityLevel()),
    m_composition(hub->getComposition()),
    m_bookmarks(NULL),
    m_content_style(m_composition->getContentStyle()),
    m_global_style(m_composition->getGlobalStyle()),
    m_types(m_ok->getTypes()),
    m_title(NULL),
    m_scene(NULL),
    m_frame(NULL),
    m_window(NULL),
    m_interactor(NULL),
    m_seqhis(NULL),
    m_seqmat(NULL),
    m_boundaries(NULL),
    m_photons(NULL),
    m_gui(NULL),
    m_launcher(NULL),
    m_external_renderer(NULL)
{
    init();
    (*m_log)("DONE");
}

void OpticksViz::init()
{
    OKI_PROFILE("OpticksViz::OpticksViz"); 

    m_hub->setCtrl(this);  // For "command(char* ctrl)" interface from lower levels to route via OpticksViz

    m_scene      = new Scene(m_hub) ;
    m_frame       = new Frame ; 
    m_interactor  = new Interactor(m_composition) ;  

    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    //m_interactor->setComposition(m_composition);

    m_scene->setInteractor(m_interactor);      

    m_frame->setInteractor(m_interactor);      
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_hub->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
    m_hub->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
    m_hub->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));

    if(m_immediate)
    {
        
        m_hub->configureState(getSceneConfigurable()) ;    // loads/creates Bookmarks
        m_bookmarks = m_hub->getBookmarks();

        const char* renderMode = m_ok->getRenderMode();

        LOG(LEVEL) << "renderMode " << renderMode ; 

        prepareScene(renderMode);      // setup OpenGL shaders and creates OpenGL context (the window)
 
        uploadGeometry();    // Scene::uploadGeometry, hands geometry to the Renderer instances for upload


        const char* renderCmd = m_ok->getRenderCmd(); 
        commandline(renderCmd); 

    }
}


void OpticksViz::visualize()
{
    prepareGUI();
    renderLoop();
}



void OpticksViz::setTitle(const char* title)
{
    m_title = title ? strdup(title) : NULL ; 
}
void OpticksViz::setLauncher(SLauncher* launcher)
{
    m_launcher = launcher ; 
}
Scene* OpticksViz::getScene()
{
    return m_scene ; 
}
Opticks* OpticksViz::getOpticks()
{
    return m_ok ; 
}
Interactor* OpticksViz::getInteractor()
{
    return m_interactor ; 
}
OpticksHub* OpticksViz::getHub()
{
    return m_hub ; 
}
NConfigurable* OpticksViz::getSceneConfigurable()
{
    return dynamic_cast<NConfigurable*>(m_scene) ; 
}

/**
OpticksViz::command
----------------------

Commands currently only from InterpolatedView flightpaths

**/

void OpticksViz::command(const char* cmd)
{
    assert( strlen(cmd) == 2 ); 
    LOG(LEVEL) << cmd ; 

    // m_hub->command(cmd); // hub always redirects to composition anyhow
    m_composition->command(cmd) ;    
    m_scene->command(cmd);  // some commands need explicit scene updating 
}

void OpticksViz::commandline(const char* cmdline)
{
    if(cmdline == NULL) return ; 

    std::vector<std::string> cmds ;
    BStr::split(cmds, cmdline, ',' );

    LOG(LEVEL) 
         << " cmdline " << cmdline 
         << " cmds " << cmds.size()
         ;

    for(unsigned i=0 ; i < cmds.size() ; i++) 
    {
        std::string& cmd = cmds[i] ; 
        command(cmd.c_str()); 
    }
}




bool OpticksViz::hasOpt(const char* name)
{
    return m_hub->hasOpt(name);
}

void OpticksViz::setupRendermode(const char* rendermode )
{
    LOG(LEVEL) << "[" << ( rendermode ? rendermode : "-" ) << "]"  ;

    if(rendermode)
    { 
        LOG(LEVEL) << "using non-standard rendermode " << rendermode ;
        m_scene->setRenderMode(rendermode);
    } 
    else
    { 
        if(m_ok->isJuno())  // hmm: dirty, can such stuff "default argument setup" be done at bash level
        {
            m_scene->setRenderMode("bb0,bb1,-global");
        }
    }
    std::string rmode = m_scene->getRenderMode();
    LOG(LEVEL) << " rmode " << rmode ; 

    m_scene->setInstCull( m_ok->hasOpt("instcull" ) );

}

void OpticksViz::setupRestrictions()
{
    if(m_ok->isJuno())
    {
        LOG(error) << "disable GeometryStyle  WIRE for JUNO as too slow " ;

        if(!hasOpt("jwire")) // use --jwire to enable wireframe with JUNO, do this only on workstations with very recent GPUs
        { 
            m_content_style->setNumContentStyle(ContentStyle::WIRE); 
        }
        m_global_style->setNumGlobalStyle(GlobalStyle::GVISVEC); // disable GVISVEC, GVEC debug styles
    }
    else if(m_ok->isDayabay())
    {
        m_global_style->setNumGlobalStyle(GlobalStyle::GVISVEC);   // disable GVISVEC, GVEC debug styles
    }
}


void OpticksViz::prepareScene(const char* rendermode)
{
    setupRendermode(rendermode);
    setupRestrictions();

    BDynamicDefine* dd = m_ok->makeDynamicDefine(); 
    m_scene->write(dd);

    if(m_hub->hasOpt("dbginterop"))
    {
        m_scene->initRenderersDebug();  // reading shader source and creating subset of renderers
    }
    else
    {
        m_scene->initRenderers();  // reading shader source and creating renderers
    }

    m_scene->setRecordStyle( m_hub->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    

    m_frame->setTitle(m_title ? m_title : "OpticksViz");
    m_frame->setFullscreen(hasOpt("fullscreen"));
    m_frame->init();           // creates OpenGL context

    m_window = m_frame->getWindow();


    //m_scene->setComposition(m_hub->getComposition());   
    m_scene->hookupRenderers();   // deferred until here, after renderers are setup 

  
}



void OpticksViz::uploadGeometry()
{
    LOG(LEVEL) << "[ hub " << m_hub->desc() ;

    NPY<unsigned char>* colors = m_hub->getColorBuffer();

    m_scene->uploadColorBuffer( colors );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"

    LOG(info) << m_ok->description();

    m_composition->setTimeDomain(        m_ok->getTimeDomain() );
    m_composition->setDomainCenterExtent(m_ok->getSpaceDomain());

    //m_scene->setGeometry(m_ggb->getGeoLib());
    m_scene->setGeometry(m_ggeo);

    m_scene->uploadGeometry();


    m_hub->setupCompositionTargetting();

    LOG(LEVEL) << "]" ;
}

int OpticksViz::getTarget()
{   
   return m_hub->getTarget() ; 
}


void OpticksViz::uploadEvent()
{
    if(m_hub->hasOpt("nooptix|noevent")) return ; 
 
    m_composition->update();

    OpticksEvent* evt = m_run->getCurrentEvent() ;

    uploadEvent(evt);
}

void OpticksViz::uploadEvent(OpticksEvent* evt)
{
    LOG(LEVEL) << "[ (" << evt->getId() << ")" ;

    m_scene->upload(evt);

    if(m_hub->hasOpt("dbguploads"))
        m_scene->dump_uploads_table("OpticksViz::uploadEvent(--dbguploads)");

    LOG(LEVEL) << "] (" << evt->getId() << ")" ;
}




void OpticksViz::downloadData(NPY<float>* data)
{
    assert(data);
    LOG(LEVEL) ;
    Rdr::download<float>(data);
}

void OpticksViz::downloadEvent()
{
    OpticksEvent* evt = m_run->getEvent();  // almost always OK evt never G4 ?
    assert(evt);
    LOG(LEVEL) << "[ (" << evt->getId() << ")" ;
    Rdr::download(evt);
    LOG(LEVEL) << "] (" << evt->getId() << ")" ;
}


void OpticksViz::indexPresentationPrep()
{
    if(!m_idx) return ; 

    LOG(LEVEL) ; 

    m_seqhis = m_idx->makeHistoryItemIndex();
    m_seqmat = m_idx->makeMaterialItemIndex();
    m_boundaries = m_idx->makeBoundaryItemIndex();

}

void OpticksViz::prepareGUI()
{
    Bookmarks* bookmarks=m_hub->getBookmarks();

    bookmarks->create(0);

#ifdef GUI_

    if(m_idx)
    {
        Types* types = m_ok->getTypes();  // needed for each render
        m_photons = new Photons(types, m_boundaries, m_seqhis, m_seqmat ) ; // GUI jacket 
        m_scene->setPhotons(m_photons);
    }

    m_gui = new GUI(m_hub) ;
    m_gui->setScene(m_scene);
    m_gui->setPhotons(m_photons);
    m_gui->setComposition(m_hub->getComposition());
    m_gui->setBookmarks(bookmarks);
    m_gui->setStateGUI(new StateGUI(m_hub->getState()));
    m_gui->setInteractor(m_interactor);   // status line
    
    m_gui->init(m_window);
    m_gui->setupHelpText( m_hub->getCfgString() );

    OpticksEvent* evt = m_run->getCurrentEvent();


    OpticksProfile* profile = evt ? evt->getProfile() : NULL ; 
    if(profile)
    {
        m_gui->setupStats(profile->getLines());
    }
    else
    {
        LOG(LEVEL) << "NULL profile " ; 
    }  

    NMeta* parameters = evt ? evt->getParameters() : m_ok->getParameters() ; 
    m_gui->setupParams(parameters->getLines());

#endif

}



void OpticksViz::renderGUI()
{
#ifdef GUI_
    if(!m_gui) return ; 
    m_gui->newframe();
    bool* show_gui_window = m_interactor->getGUIModeAddress();
    if(*show_gui_window)
    {
        m_gui->show(show_gui_window);
        if(m_photons)
        {
            if(m_boundaries)
            {
                m_composition->getPick().y = m_boundaries->getSelected() ;   //  1st boundary 
            }
            glm::ivec4& recsel = m_composition->getRecSelect();
            recsel.x = m_seqhis ? m_seqhis->getSelected() : 0 ; 
            recsel.y = m_seqmat ? m_seqmat->getSelected() : 0 ; 
            m_composition->setFlags(m_types->getFlags());      // TODO: check why this is here ?
        }
        // maybe imgui edit selection within the composition imgui, rather than shovelling ?
        // BUT: composition feeds into shader uniforms which could be reused by multiple classes ?
    }

    bool* show_scrub_window = m_interactor->getScrubModeAddress();
    if(*show_scrub_window)
        m_gui->show_scrubber(show_scrub_window);


    bool* show_label_window = m_interactor->getLabelModeAddress();
    if(*show_label_window)
        m_gui->show_label(show_label_window);


    m_gui->render();
#endif
}


void OpticksViz::setExternalRenderer(SRenderer* external_renderer)
{
    m_external_renderer = external_renderer ; 
}

void OpticksViz::render()
{
    m_frame->viewport();
    m_frame->clear();

    if(m_composition->isRaytracedRender() || m_composition->isCompositeRender()) 
    {
        if(m_external_renderer) m_external_renderer->render();
    }

    m_scene->render();
}





void OpticksViz::renderLoop()
{
    if(m_interactivity == 0 )
    {
        LOG(LEVEL) << "early exit due to InteractivityLevel 0  " ; 
        return ;
    }
    LOG(LEVEL) << "enter runloop "; 

    //m_frame->toggleFullscreen(true); causing blankscreen then segv
    m_frame->hintVisible(true);
    m_frame->show();
    LOG(LEVEL) << "after frame.show() "; 

    unsigned count(0) ; 
    bool exitloop(false); 

    int renderlooplimit = m_ok->getRenderLoopLimit(); 

    while (!glfwWindowShouldClose(m_window) && !exitloop  )
    {
        m_frame->listen(); 

#ifdef OPTICKS_NPYSERVER
        if(m_server) m_server->poll_one();  
#endif
        count = m_composition->tick();

        if(m_launcher)
        {
            m_launcher->launch(count);
        }

        if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)  
        {
            render();
            renderGUI();

            glfwSwapBuffers(m_window);

            m_interactor->setChanged(false);  
            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }

        exitloop = renderlooplimit > 0 && int(count) > renderlooplimit ; 
    }

    LOG(LEVEL) 
        << " renderlooplimit " << renderlooplimit 
        << " count " << count 
        ;

}


void OpticksViz::cleanup()
{
#ifdef GUI_
    if(m_gui) m_gui->shutdown();
#endif
    if(m_frame) m_frame->exit();
}


 
