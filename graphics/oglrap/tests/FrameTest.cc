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

    frame.setSize(640,480);
    frame.setTitle("FrameTest");
    frame.init_window();

    Texture texture ;
    texture.create(frame.getWidth(), frame.getHeight()) ; 
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
