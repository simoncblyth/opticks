#include <stdlib.h>  //exit()
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLEQ_IMPLEMENTATION
#include "gleq.h"

#include "app.hh"
#include "AppCfg.hh"

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

#include "numpydelegate.hh"
#include "numpydelegateCfg.hh"
#include "numpyserver.hpp"



int main(int argc, char** argv)
{
    App app ;  // misnomer : more like window Frame
    numpydelegate delegate ; 
    Scene scene ;  // just instanciates Camera and View, allowing config hookup 
    Interactor interactor ; 

    interactor.setScene(&scene);
    app.setScene(&scene);
    app.setInteractor(&interactor);  // TODO: decide on who contains who

    AppCfg<App>* appcfg = new AppCfg<App>("app", &app, false);

    Cfg cfg("unbrella", false) ;  // collect other Cfg objects
    cfg.add(appcfg);
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));
    cfg.add(new SceneCfg<Scene>("scene", &scene, true));
    cfg.add(new CameraCfg<Camera>("camera", scene.getCamera(), true));
    cfg.add(new ViewCfg<View>(    "view",   scene.getView(),   true));
    cfg.add(new TrackballCfg<Trackball>( "trackball",   scene.getTrackball(),   true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,   true));

    cfg.commandline(argc, argv);
    delegate.liveConnect(&cfg);    

    if(appcfg->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(appcfg->isAbort()) exit(EXIT_SUCCESS); 

    numpyserver<numpydelegate> srv(&delegate);

    app.setSize(640,480);
    app.setTitle("Demo");
    app.init_window();

    scene.load("GLFWTEST_") ;
    scene.init_opengl();

    GLFWwindow* window = app.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        app.listen(); 

        // give numpyserver a few cycles, to complete posts from the net thread
        // resulting in the non-blocking handler methods of the delegate being called
        srv.poll_one();  

        app.render();
        glfwSwapBuffers(window);
    }
    srv.stop();


    app.exit();

    exit(EXIT_SUCCESS);
}

