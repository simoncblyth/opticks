#include "Composition.hh"
#include "OpticksHub.hh"
#include "OpticksViz.hh"

// opticksgeo-
#include "OpticksHub.hh"

// windows headers from PLOG need to be before glfw 
// http://stackoverflow.com/questions/3927810/how-to-prevent-macro-redefinition
#include "PLOG.hh"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"

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


OpticksViz::OpticksViz(OpticksHub* hub)
    :
    m_opticks(hub->getOpticks()),
    m_hub(hub),
    m_scene(NULL),
    m_composition(hub->getComposition()),
    m_frame(NULL),
    m_window(NULL),
    m_interactor(NULL)
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


void OpticksViz::prepareScene()
{
    m_scene->setEvent(m_hub->getEvent());

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

  
