#include "Frame.hh"
#include "Composition.hh"
#include "Renderer.hh"
#include "Interactor.hh"
#include "Texture.hh"

int main(int argc, char** argv)
{
    if(argc < 2)  
    {   
        printf("%s : expecting argument with path to ppm file\n", argv[0]);
        return 1;  
    }   
    char* ppmpath = argv[1] ;

    Frame frame ; 
    Composition composition ; 
    Interactor interactor ;
    Renderer renderer("tex") ; 

    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    interactor.setComposition(&composition); // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);  // composition provides matrices to renderer 



    Texture texture ;
    texture.loadPPM(ppmpath);
    composition.setSize(texture.getWidth(), texture.getHeight(),2 );

    frame.setComposition(&composition);
    frame.setTitle("FrameTest");
    frame.init(); // creates OpenGL context 

    //composition.setModelToWorld(texture.getModelToWorldPtr(0));   // point at the geometry 
    composition.setCenterExtent(texture.getCenterExtent(0));   // point at the geometry 
    composition.update();
    composition.Details("Composition::details");

    texture.create();   // after OpenGL context creation, done in frame.gl_init_window
    renderer.setDrawable(&texture);

    GLFWwindow* window = frame.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        frame.listen();
        frame.render();
        composition.update();

        renderer.render();
        glfwSwapBuffers(window);
    }
    frame.exit();

    return 0 ;
}
