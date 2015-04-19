#include <stdlib.h>  //exit()
#include <stdio.h>

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Composition.hh"
#include "CompositionCfg.hh"
#include "Frame.hh"
#include "FrameCfg.hh"
#include "Geometry.hh"

#include "Scene.hh"
#include "Rdr.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"

#include "Interactor.hh"
#include "InteractorCfg.hh"
#include "Camera.hh"
#include "CameraCfg.hh"
#include "View.hh"
#include "ViewCfg.hh"
#include "Trackball.hh"
#include "TrackballCfg.hh"
#include "Clipper.hh"
#include "ClipperCfg.hh"
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
    Interactor interactor ; 
    numpydelegate delegate ; 

    NumpyEvt evt ;
    evt.setGenstepData(NPY::load("cerenkov", "1")); 

    Scene scene ;
    scene.setNumpyEvt(&evt);
    scene.setComposition(&composition);    

    Cfg cfg("umbrella", false) ;             // collect other Cfg objects
    cfg.add(new FrameCfg<Frame>("frame", &frame, false));
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));
    cfg.add(new RendererCfg<Renderer>("renderer", scene.getGeometryRenderer(), true));
    cfg.add(new CompositionCfg<Composition>("composition", &composition, true));
    cfg.add(new CameraCfg<Camera>("camera", composition.getCamera(), true));
    cfg.add(new ViewCfg<View>(    "view",   composition.getView(),   true));
    cfg.add(new TrackballCfg<Trackball>( "trackball",   composition.getTrackball(),   true));
    cfg.add(new ClipperCfg<Clipper>( "clipper",   composition.getClipper(),   true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,   true));

    cfg.commandline(argc, argv);
    delegate.liveConnect(&cfg);     
    delegate.setNumpyEvt(&evt);

    if(cfg["frame"]->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(cfg["frame"]->isAbort()) exit(EXIT_SUCCESS); 

    numpyserver<numpydelegate> server(&delegate); // connect to external messages 
    frame.setInteractor(&interactor);             // GLFW key/mouse events from frame to interactor and on to composition constituents
    interactor.setup(composition.getCamera(), composition.getView(), composition.getTrackball(), composition.getClipper());  


    frame.gl_init_window("GGeoView", composition.getWidth(),composition.getHeight());    // creates OpenGL context 

    scene.loadGeometry("GGEOVIEW_") ; 
    composition.setModelToWorld(scene.getTarget());
    scene.loadEvt();

    //
    // TODO:  
    //   * pull out the OptiX engine renderer to be external, and fit in with the scene ?
    //   * extract core OptiX processing into separate class
    //   * hmm generation should not depend on renderers OpenGL buffers
    //     but for OpenGL interop its expedient for now
    //
    OptiXEngine engine("GGeoView") ;       
    Geometry* geoloader = scene.getGeometryLoader();     // needing both GGeo and GMergedMesh is transitional
    engine.setGGeo(geoloader->getGGeo());            
    engine.setMergedMesh(geoloader->getMergedMesh());
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

