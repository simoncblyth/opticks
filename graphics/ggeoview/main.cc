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

#include "OptiXEngine.hh"
#include "RayTraceConfig.hh"

int main(int argc, char** argv)
{
    Frame frame ;
    Composition composition ;   
    Interactor interactor ; 
    Renderer renderer ;  
    Geometry geometry ;
    numpydelegate delegate ; 

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

    if(cfg["frame"]->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(cfg["frame"]->isAbort()) exit(EXIT_SUCCESS); 

    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    interactor.setup(composition.getCamera(), composition.getView(), composition.getTrackball());  // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);
    numpyserver<numpydelegate> server(&delegate);

    frame.gl_init_window("GGeoView", composition.getWidth(),composition.getHeight()); // OpenGL context created
    geometry.load("GGEOVIEW_") ; 
    renderer.setDrawable(geometry.getDrawable());

    // to see the geometry normally need diffeerent shader without tex

    //OptiXEngine engine("GGeoView") ;        // creates OptiX context
    //engine.setComposition(&composition); 
    //engine.initContext();
    //engine.associate_PBO_to_Texture(texture.getTextureId());    // teapot replaced with plain grey
    //engine.preprocess();

    GLFWwindow* window = frame.getWindow();
    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 
        server.poll_one();  
        //engine.trace();
        //engine.displayFrame(texture.getTextureId());
        frame.render();
        renderer.render();
        glfwSwapBuffers(window);
    }
    //server.sleep(10);
    server.stop();
    frame.exit();
    exit(EXIT_SUCCESS);
}

