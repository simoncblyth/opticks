#include "Frame.hh"
#include "Scene.hh"
#include "Interactor.hh"
#include "Texture.hh"

int main()
{
    Frame frame ; 
    Scene scene ; 
    Interactor interactor ;

    frame.setInteractor(&interactor);  // needed for the GLFW key and mouse events to be funneled to Interactor
    interactor.setScene(&scene);       // allows interactor to modify camera, view and trackball constituents of scene
    frame.setScene(&scene);            // only so frame.render() calls draw on scene  

    Texture texture ;
    texture.loadPPM("/tmp/teapot.ppm");

    frame.setSize(texture.getWidth(),texture.getHeight());
    frame.setTitle("FrameTest");
    frame.init_window();

    texture.create();   // needs to be after OpenGL context creation, done in frame.init_window

    scene.setGeometry(&texture);
    scene.init_opengl();              // uploads geometry buffers

    GLFWwindow* window = frame.getWindow();

    while (!glfwWindowShouldClose(window))
    {
        frame.listen();
        frame.render();
        glfwSwapBuffers(window);
    }
    frame.exit();

    return 0 ;
}
