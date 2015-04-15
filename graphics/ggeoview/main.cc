#include <stdlib.h>  //exit()
#include <stdio.h>

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Composition.hh"
#include "CompositionCfg.hh"
#include "Frame.hh"
#include "FrameCfg.hh"
#include "Geometry.hh"
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
#include "Texture.hh"

// numpyserver-
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#include "NumpyEvt.hpp"


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
    Renderer renderer("nrm") ;  
    Geometry geometry ;
    numpydelegate delegate ; 
    NumpyEvt evt ; 

    Cfg cfg("umbrella", false) ;             // collect other Cfg objects
    cfg.add(new FrameCfg<Frame>("frame", &frame, false));
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));
    cfg.add(new RendererCfg<Renderer>("renderer", &renderer, true));
    cfg.add(new CompositionCfg<Composition>("composition", &composition, true));
    cfg.add(new CameraCfg<Camera>("camera", composition.getCamera(), true));
    cfg.add(new ViewCfg<View>(    "view",   composition.getView(),   true));
    cfg.add(new TrackballCfg<Trackball>( "trackball",   composition.getTrackball(),   true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,   true));

    cfg.commandline(argc, argv);
    delegate.liveConnect(&cfg);     
    delegate.setNumpyEvt(&evt);

    evt.setNPY(NPY::load("cerenkov", "1"));  // for dev avoid having to use npysend.sh and zmq-broker
    LOG(info) << evt.description("main/NumpyEvt") ; 


    if(cfg["frame"]->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(cfg["frame"]->isAbort()) exit(EXIT_SUCCESS); 

    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    interactor.setup(composition.getCamera(), composition.getView(), composition.getTrackball());  // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);    // renderer needs access to view matrices
    numpyserver<numpydelegate> server(&delegate);

    frame.gl_init_window("GGeoView", composition.getWidth(),composition.getHeight()); // creates OpenGL context 
    geometry.load("GGEOVIEW_") ; 
    renderer.setDrawable(geometry.getDrawable());


    OptiXEngine engine("GGeoView") ;       
    // needing both is transitional
    engine.setGGeo(geometry.getGGeo());
    engine.setMergedMesh(geometry.getMergedMesh());
    engine.setComposition(&composition);   // engine needs access to the view matrices
    engine.setEnabled(interactor.getOptiXMode()>-1);
    engine.init();    // creates OptiX context, when enabled
 
    GLFWwindow* window = frame.getWindow();
    LOG(info) << "enter runloop" ; 
    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 
        server.poll_one();  
        frame.render();

        //if(evt.hasNPY()) LOG(info)<< evt.description("NumpyEvt");

        if(interactor.getOptiXMode()>0)
        { 
            engine.trace();
            engine.render();
        }
        else
        {
            renderer.render();
        }
        glfwSwapBuffers(window);
    }
    engine.cleanUp();
    server.stop();
    frame.exit();
    exit(EXIT_SUCCESS);
}

