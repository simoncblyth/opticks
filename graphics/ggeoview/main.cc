#include <stdlib.h>  //exit()
#include <stdio.h>

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
#include "FrameCfg.hh"
#include "Scene.hh"
#include "SceneCfg.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"
#include "Interactor.hh"
#include "InteractorCfg.hh"

#include "Bookmarks.hh"
#include "Composition.hh"
#include "Geometry.hh"
#include "Rdr.hh"
#include "Texture.hh"

// numpyserver-
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "NumpyEvt.hpp"
#include "VecNPY.hpp"
#include "MultiVecNPY.hpp"


#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// optixrap-
#include "OptiXEngine.hh"
#include "RayTraceConfig.hh"


// ggeo-
#include "GMergedMesh.hh"



#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

void logging_init()
{
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );
}

int main(int argc, char** argv)
{
    logging_init();
    LOG(info) << argv[0] ; 

    Frame frame ;
    Composition composition ;   
    Bookmarks bookmarks ; 
    Interactor interactor ; 
    numpydelegate delegate ; 

    composition.setPixelFactor(2); // 2: makes OptiX render at retina resolution
    frame.setInteractor(&interactor);             // GLFW key/mouse events from frame to interactor and on to composition constituents
    interactor.setComposition(&composition);
    interactor.setBookmarks(&bookmarks);

    NumpyEvt evt ;
    evt.setGenstepData(NPY::load("cerenkov", "1")); 

    Scene scene ;
    scene.setNumpyEvt(&evt);
    scene.setComposition(&composition);    

    bookmarks.setScene(&scene);
    bookmarks.setComposition(&composition);

    Cfg cfg("umbrella", false) ; // collect other Cfg objects
    cfg.add(new FrameCfg<Frame>(                "frame",         &frame,    false));
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));

    cfg.add(new SceneCfg<Scene>(           "scene",       &scene,                      true));
    cfg.add(new RendererCfg<Renderer>(     "renderer",    scene.getGeometryRenderer(), true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,                 true));
    composition.addConfig(&cfg); 

    cfg.commandline(argc, argv);
    delegate.liveConnect(&cfg); // hookup live config via UDP messages
    delegate.setNumpyEvt(&evt); // allows delegate to update evt when NPY messages arrive

    if(cfg["frame"]->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(cfg["frame"]->isAbort()) exit(EXIT_SUCCESS); 

    numpyserver<numpydelegate> server(&delegate); // connect to external messages 

    frame.gl_init_window("GGeoView", composition.getWidth(),composition.getHeight());    // creates OpenGL context 


    bookmarks.load("/tmp/bookmarks.ini"); // hmm need to tie bookmarks with the geomety 
    scene.loadGeometry("GGEOVIEW_") ; 
    scene.loadEvt();

    //
    // TODO:  
    //   * pull out the OptiX engine renderer to be external, and fit in with the scene ?
    //   * extract core OptiX processing into separate class
    //   * hmm generation should not depend on renderers OpenGL buffers
    //     but for OpenGL interop its expedient for now
    //
    OptiXEngine engine("GGeoView") ;       

    engine.setMergedMesh(scene.getMergedMesh()); // aiming for all geo info to come from GMergedMesh
    engine.setGGeo(scene.getGGeo());             // need for GGeo too is transitional, until sort out material/surface property buffers

    engine.setComposition(&composition);                 
    engine.setEnabled(interactor.getOptiXMode()>-1);
    engine.init();                                        // creates OptiX context, when enabled
    engine.initGenerate(&evt);
 
    GLFWwindow* window = frame.getWindow();

    LOG(info) << "enter runloop "; 
    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 
        server.poll_one();  
        frame.render();

        if(interactor.getOptiXMode()>0)
        { 
            engine.trace();
            engine.render();
        }
        else
        {
            scene.render();
        }

        glfwSwapBuffers(window);
    }
    engine.cleanUp();
    server.stop();
    frame.exit();
    exit(EXIT_SUCCESS);
}

