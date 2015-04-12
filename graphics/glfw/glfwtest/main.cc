#include <stdlib.h>  //exit()
#include <stdio.h>

// oglrap-
//  Frame include brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
#include "FrameCfg.hh"
#include "Composition.hh"
#include "CompositionCfg.hh"
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

// numpyserver-
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"


int main(int argc, char** argv)
{
    Frame frame ;
    Composition composition ;
    Interactor interactor ; 
    Renderer renderer ; 
    Geometry geometry ; 
    numpydelegate delegate ; 

    Cfg cfg("umbrella", false) ;  // collect other Cfg objects
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

    if(cfg["frame"]->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(cfg["frame"]->isAbort()) exit(EXIT_SUCCESS); 

    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    interactor.setup(composition.getCamera(), composition.getView(), composition.getTrackball());  // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);   // renderer needs access to MV, MVP matrices
    numpyserver<numpydelegate> server(&delegate);

    frame.gl_init_window("GLFWTest", composition.getWidth(),composition.getHeight());
    geometry.load("GLFWTEST_") ;
    renderer.setDrawable(geometry.getDrawable());

    GLFWwindow* window = frame.getWindow();
    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 
        server.poll_one();  
        frame.render();
        renderer.render();
        glfwSwapBuffers(window);
    }
    server.stop();
    frame.exit();
    exit(EXIT_SUCCESS);
}

