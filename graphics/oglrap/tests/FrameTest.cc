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
    Renderer renderer ; 

    frame.setInteractor(&interactor);    // GLFW key and mouse events from frame to interactor
    interactor.setup(composition.getCamera(), composition.getView(), composition.getTrackball());  // interactor changes camera, view, trackball 
    renderer.setComposition(&composition);  // composition provides matrices to renderer 


    Texture texture ;
    texture.loadPPM(ppmpath);

    frame.setTitle("FrameTest");
    frame.setSize(texture.getWidth(),texture.getHeight());
    composition.setSize(frame.getWidth(), frame.getHeight());

    frame.gl_init_window();


    texture.create();   // after OpenGL context creation, done in frame.gl_init_window
    renderer.setDrawable(&texture);

    GLFWwindow* window = frame.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        frame.listen();
        frame.render();
        renderer.render();
        glfwSwapBuffers(window);
    }
    frame.exit();

    return 0 ;
}
