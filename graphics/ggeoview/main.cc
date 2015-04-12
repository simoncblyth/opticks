#include <stdlib.h>  //exit()
#include <stdio.h>

// oglrap-
//  Frame include brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Composition.hh"
#include "CompositionCfg.hh"
#include "Frame.hh"
#include "FrameCfg.hh"
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

    numpydelegate delegate ; 

    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    interactor.setup(composition.getCamera(), composition.getView(), composition.getTrackball());  // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);

    // hmm texture is detail of the renderer, should not be here
    Texture texture ;
    {
        const char* path = "/tmp/teapot.ppm" ;
        texture.loadPPM(path);
        composition.setSize(texture.getWidth(),texture.getHeight());
    }

    FrameCfg<Frame>* framecfg = new FrameCfg<Frame>("frame", &frame, false);

    Cfg cfg("unbrella", false) ;  // collect other Cfg objects
    cfg.add(framecfg);
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));
    cfg.add(new RendererCfg<Renderer>("renderer", &renderer, true));
    cfg.add(new CompositionCfg<Composition>("composition", &composition, true));
    cfg.add(new CameraCfg<Camera>("camera", composition.getCamera(), true));
    cfg.add(new ViewCfg<View>(    "view",   composition.getView(),   true));
    cfg.add(new TrackballCfg<Trackball>( "trackball",   composition.getTrackball(),   true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,   true));

    cfg.commandline(argc, argv);
    delegate.liveConnect(&cfg);    

    // hmm these below elswhere, as are needed for non-GUI apps too
    if(framecfg->isHelp())  std::cout << cfg.getDesc() << std::endl ;
    if(framecfg->isAbort()) exit(EXIT_SUCCESS); 

    numpyserver<numpydelegate> srv(&delegate);

    frame.setSize(composition.getWidth(),composition.getHeight());
    frame.setTitle("Demo");
    frame.gl_init_window();  // OpenGL context created


    //renderer.load("GGEOVIEW_") ;  // envvar prefixes
    //renderer.gl_upload_buffers();

    //texture.setSize(width, height);  incorrect to setSize when texture loaded from PPM 
    texture.create();


    // logically OptiX setup should not come after OpenGL stuff (as need for non-GUI running) 
    // but when using VBO initContext needs to be after  OpenGL context creation
    // otherwidth segfaults at glGenBuffers(1, &vbo);
    //

    OptiXEngine engine ;        // creates OptiX context
    RayTraceConfig::makeInstance(engine.getContext(), "GGeoView");
    engine.initContext(composition.getWidth(), composition.getHeight());  

    engine.associate_PBO_to_Texture(texture.getTextureId());    // teapot replaced with plain grey

    renderer.setDrawable(&texture);  

    engine.preprocess();

    GLFWwindow* window = frame.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 
        srv.poll_one();  

        engine.trace();
        //engine.displayFrame(texture.getTextureId());

        frame.render();
        renderer.render();

        glfwSwapBuffers(window);
    }

    //srv.sleep(10);
    srv.stop();
    frame.exit();

    exit(EXIT_SUCCESS);
}

