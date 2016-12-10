/*
Search for a .ppm image file on your system and pass to FrameTest 

  SHADER_DIR=$(oglrap-sdir)/gl FrameTest /opt/local/lib/tk8.6/demos/images/teapot.ppm

*/

#include "NGLM.hpp"

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "Frame.hh"
#include "Composition.hh"
#include "Renderer.hh"
#include "Interactor.hh"
#include "Texture.hh"

#include "OGLRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OGLRAP_LOG__ ; 

    if(argc < 2)  
    {   
        printf("%s : expecting argument with path to ppm file\n", argv[0]);
        return 0 ;  
    }   
    char* ppmpath = argv[1] ;

    LOG(info) << argv[0] << " ppmpath " << ppmpath ; 



    Opticks ok(argc, argv);
    OpticksHub hub(&ok);


    Frame frame(&ok) ; 
    Composition composition ; 
    Interactor interactor(&hub) ;
    Renderer renderer("tex") ; 

    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    //interactor.setComposition(&composition); // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);  // composition provides matrices to renderer 


    Texture texture ;
    texture.loadPPM(ppmpath);
    composition.setSize(texture.getWidth(), texture.getHeight(),2 );

    frame.setComposition(&composition);
    frame.setTitle("FrameTest");
    frame.init(); // creates OpenGL context 

    //composition.setModelToWorld(texture.getModelToWorldPtr(0));   // point at the geometry 

    gfloat4 ce = texture.getCenterExtent(0);

    glm::vec4 ce_(ce.x, ce.y, ce.z, ce.w);
    composition.setCenterExtent(ce_); // point at the geometry 

    composition.update();
    composition.Details("Composition::details");

    texture.create();   // after OpenGL context creation, done in frame.gl_init_window
    renderer.upload(&texture);

    GLFWwindow* window = frame.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        frame.listen();
        frame.viewport();
        frame.clear();
        composition.update();

        renderer.render();
        glfwSwapBuffers(window);
    }
    frame.exit();

    return 0 ;
}
