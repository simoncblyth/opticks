
// sysrap-
#include "SRenderer.hh"

// npy-
#include "Types.hpp"
#include "Parameters.hpp"
#include "Timer.hpp"
#include "TimesTable.hpp"
#include "NGLM.hpp"
#include "NPY.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "Composition.hh"
#include "Bookmarks.hh"

#include "GItemIndex.hh"


// opticksgeo-
#include "OpticksHub.hh"

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


#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }


OpticksViz::OpticksViz(OpticksHub* hub)
    :
    m_hub(hub),
    m_opticks(hub->getOpticks()),
    m_interactivity(m_opticks->getInteractivityLevel()),
    m_composition(hub->getComposition()),
    m_types(m_opticks->getTypes()),
    m_scene(NULL),
    m_frame(NULL),
    m_window(NULL),
    m_interactor(NULL),
    m_seqhis(NULL),
    m_seqmat(NULL),
    m_boundaries(NULL),
    m_photons(NULL),
    m_gui(NULL),
    m_external_renderer(NULL)
{
    init();
}



void OpticksViz::init()
{

    const char* shader_dir = getenv("OPTICKS_SHADER_DIR"); 
    const char* shader_incl_path = getenv("OPTICKS_SHADER_INCL_PATH"); 
    const char* shader_dynamic_dir = getenv("OPTICKS_SHADER_DYNAMIC_DIR"); 
    // envvars normally not defined, using cmake configure_file values instead

    m_scene      = new Scene(shader_dir, shader_incl_path, shader_dynamic_dir ) ;
    m_frame       = new Frame ; 
    m_interactor  = new Interactor ; 

    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    m_interactor->setComposition(m_composition);

    m_scene->setInteractor(m_interactor);      

    m_frame->setInteractor(m_interactor);      
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_hub->add(new SceneCfg<Scene>(           "scene",       m_scene,                      true));
    m_hub->add(new RendererCfg<Renderer>(     "renderer",    m_scene->getGeometryRenderer(), true));
    m_hub->add(new InteractorCfg<Interactor>( "interactor",  m_interactor,                 true));

}

void OpticksViz::setEvent(OpticksEvent* evt)
{
    m_scene->setEvent(evt);
}
OpticksEvent* OpticksViz::getEvent()
{
    return m_scene->getEvent();
}



Scene* OpticksViz::getScene()
{
    return m_scene ; 
}

bool OpticksViz::hasOpt(const char* name)
{
    return m_hub->hasOpt(name);
}

void OpticksViz::configure()
{
    m_hub->configureViz(m_scene) ;
    m_interactor->setBookmarks(m_hub->getBookmarks());
}


void OpticksViz::prepareScene()
{
    if(m_opticks->isJuno())
    {
        LOG(warning) << "App::prepareViz disable GeometryStyle  WIRE for JUNO as too slow " ;

        if(!hasOpt("jwire")) // use --jwire to enable wireframe with JUNO, do this only on workstations with very recent GPUs
        { 
            m_scene->setNumGeometryStyle(Scene::WIRE); 
        }

        m_scene->setNumGlobalStyle(Scene::GVISVEC); // disable GVISVEC, GVEC debug styles

        m_scene->setRenderMode("bb0,bb1,-global");
        std::string rmode = m_scene->getRenderMode();
        LOG(info) << "App::prepareViz " << rmode ; 
    }
    else if(m_opticks->isDayabay())
    {
        m_scene->setNumGlobalStyle(Scene::GVISVEC);   // disable GVISVEC, GVEC debug styles
    }


    BDynamicDefine* dd = m_opticks->makeDynamicDefine(); 
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


    m_frame->setTitle("GGeoView");
    m_frame->setFullscreen(hasOpt("fullscreen"));
    m_frame->init();           // creates OpenGL context

    m_window = m_frame->getWindow();

    m_scene->setComposition(m_hub->getComposition());     // deferred until here, after renderers are setup 

}



void OpticksViz::uploadGeometry()
{
    NPY<unsigned char>* colors = m_hub->getColorBuffer();

    m_scene->uploadColorBuffer( colors );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"

    // where does this info come from, where is most appropriate to set it ?
    // domains are set on running Opticks::makeEvent 
    //  BUT the space domain has to be updated after geometry loaded 

    
    LOG(fatal) << "OpticksViz::uploadGeometry opticks domains " << m_opticks->description();

    m_composition->setTimeDomain(        m_opticks->getTimeDomain() );
    m_composition->setDomainCenterExtent(m_opticks->getSpaceDomain());

    m_scene->setGeometry(m_hub->getGGeo());

    m_scene->uploadGeometry();

    bool autocam = true ; 

    // handle commandline --target option that needs loaded geometry 
    unsigned int target = m_scene->getTargetDeferred();   // default to 0 
    LOG(debug) << "App::uploadGeometryViz setting target " << target ; 

    m_scene->setTarget(target, autocam);

}

void OpticksViz::targetGenstep()
{
    if(m_scene->getTarget() == 0) // only target based on genstep if not already targetted
    {
        m_hub->targetGenstep();
    }
}

void OpticksViz::uploadEvent()
{
    if(m_hub->hasOpt("nooptix|noevent")) 
    {
        LOG(warning) << "OpticksViz::uploadEvent skip due to --nooptix/--noevent " ;
        return ;
    }
 
    LOG(info) << "OpticksViz::uploadEvent START " ;

    m_composition->update();

    m_scene->upload();

    m_scene->uploadSelection();

    if(m_hub->hasOpt("dbguploads"))
        m_scene->dump_uploads_table("OpticksViz::uploadEvent");

}



void OpticksViz::downloadData(NPY<float>* data)
{
    assert(data);
    LOG(info) << "OpticksViz::downloadData" ;
    Rdr::download<float>(data);
}

void OpticksViz::downloadEvent()
{
    OpticksEvent* evt = getEvent(); 
    assert(evt);

    LOG(info) << "OpticksViz::downloadEvent" ;
    Rdr::download(evt);
}


void OpticksViz::indexPresentationPrep()
{
    LOG(info) << "OpticksViz::indexPresentationPrep" ; 

    m_seqhis = m_hub->makeHistoryItemIndex();
    m_seqmat = m_hub->makeMaterialItemIndex();
    m_boundaries = m_hub->makeBoundaryItemIndex();

    TIMER("indexPresentationPrep"); 
}

void OpticksViz::prepareGUI()
{
    Bookmarks* bookmarks=m_hub->getBookmarks();

    bookmarks->create(0);

#ifdef GUI_

    Types* types = m_opticks->getTypes();  // needed for each render
    m_photons = new Photons(types, m_boundaries, m_seqhis, m_seqmat ) ; // GUI jacket 
    m_scene->setPhotons(m_photons);

    m_gui = new GUI(m_hub->getGGeo()) ;
    m_gui->setScene(m_scene);
    m_gui->setPhotons(m_photons);
    m_gui->setComposition(m_hub->getComposition());
    m_gui->setBookmarks(bookmarks);
    m_gui->setStateGUI(new StateGUI(m_hub->getState()));
    m_gui->setInteractor(m_interactor);   // status line
    
    m_gui->init(m_window);
    m_gui->setupHelpText( m_hub->getCfgString() );

    OpticksEvent* evt = m_hub->getEvent();

    TimesTable* tt = evt ? evt->getTimesTable() : NULL ; 
    if(tt)
    {
        m_gui->setupStats(tt->getLines());
    }
    else
    {
        LOG(warning) << "App::prepareGUI NULL TimesTable " ; 
    }  

    Parameters* parameters = evt ? evt->getParameters() : m_opticks->getParameters() ; 

    m_gui->setupParams(parameters->getLines());

#endif

    TIMER("prepareGUI"); 
}



void OpticksViz::renderGUI()
{
#ifdef GUI_
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

    if(m_scene->isRaytracedRender() || m_scene->isCompositeRender()) 
    {
        if(m_external_renderer) m_external_renderer->render();
    }

    m_scene->render();
}



void OpticksViz::renderLoop()
{
    if(m_interactivity == 0 )
    {
        LOG(info) << "OpticksViz::renderLoop early exit due to InteractivityLevel 0  " ; 
        return ;
    }
    LOG(info) << "enter runloop "; 

    //m_frame->toggleFullscreen(true); causing blankscreen then segv
    m_frame->hintVisible(true);
    m_frame->show();
    LOG(info) << "after frame.show() "; 

    unsigned int count ; 

    while (!glfwWindowShouldClose(m_window))
    {
        m_frame->listen(); 

#ifdef WITH_NPYSERVER
        if(m_server) m_server->poll_one();  
#endif
        count = m_composition->tick();

        if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)  
        {
            render();
            renderGUI();

            glfwSwapBuffers(m_window);

            m_interactor->setChanged(false);  
            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }
    }
}


void OpticksViz::cleanup()
{
#ifdef GUI_
    if(m_gui) m_gui->shutdown();
#endif
    if(m_frame) m_frame->exit();

}






 
