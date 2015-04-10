#include <stdlib.h>  //exit()
#include <stdio.h>

// oglrap-
//  Frame include brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
#include "FrameCfg.hh"
#include "Scene.hh"
#include "SceneCfg.hh"
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


#include "OptiXEngine.hh"
#include "RayTraceConfig.hh"


int main(int argc, char** argv)
{
    unsigned int width = 640 ;
    unsigned int height = 480 ;

    Frame frame ;
    numpydelegate delegate ; 
    Scene scene ;  // ctor just instanciates Camera and View for early config
    Interactor interactor ; 


    interactor.setScene(&scene);
    frame.setScene(&scene);
    frame.setInteractor(&interactor);  // TODO: decide on who contains who

    FrameCfg<Frame>* framecfg = new FrameCfg<Frame>("frame", &frame, false);

    Cfg cfg("unbrella", false) ;  // collect other Cfg objects
    cfg.add(framecfg);
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));
    cfg.add(new SceneCfg<Scene>("scene", &scene, true));
    cfg.add(new CameraCfg<Camera>("camera", scene.getCamera(), true));
    cfg.add(new ViewCfg<View>(    "view",   scene.getView(),   true));
    cfg.add(new TrackballCfg<Trackball>( "trackball",   scene.getTrackball(),   true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,   true));

    cfg.commandline(argc, argv);
    delegate.liveConnect(&cfg);    

    // hmm these below elswhere, as are needed for non-GUI apps too
    if(framecfg->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(framecfg->isAbort()) exit(EXIT_SUCCESS); 

    numpyserver<numpydelegate> srv(&delegate);

    frame.setSize(width,height);
    frame.setTitle("Demo");
    frame.init_window();

    scene.load("GGEOVIEW_") ;  // envvar prefixes
    scene.init_opengl();


    // logically OptiX setup should not come after OpenGL stuff (as need for non-GUI running) 
    // but when using VBO initContext needs to be after  OpenGL context creation
    // otherwidth segfaults at glGenBuffers(1, &vbo);
    //
    OptiXEngine engine ;   // creates OptiX context
    RayTraceConfig::makeInstance(engine.getContext(), "GGeoView");
    engine.initContext(width, height); 






    GLFWwindow* window = frame.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 

        // give numpyserver a few cycles, to complete posts from the net thread
        // resulting in the non-blocking handler methods of the delegate being called
        srv.poll_one();  

        frame.render();
        glfwSwapBuffers(window);
    }
    srv.stop();


    frame.exit();

    exit(EXIT_SUCCESS);
}

