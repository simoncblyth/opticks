#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>  //exit()
#include <stdio.h>

#include "Scene.hh"

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}


void _update_fps_counter (GLFWwindow* window) {
  static double previous_seconds = glfwGetTime ();
  static int frame_count;
  double current_seconds = glfwGetTime ();
  double elapsed_seconds = current_seconds - previous_seconds;
  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    double fps = (double)frame_count / elapsed_seconds;
    char tmp[128];
    sprintf (tmp, "opengl @ fps: %.2f", fps);
    glfwSetWindowTitle (window, tmp);
    frame_count = 0;
  }
  frame_count++;
}




int main(void)
{
    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

#ifdef  __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    int window_width = 640 ;
    int window_height = 480 ;
    const char* window_title = "Title" ;

    window = glfwCreateWindow(window_width, window_height, window_title, NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    // start GLEW extension handler, segfaults if done before glfwCreateWindow
    glewExperimental = GL_TRUE;
    glewInit ();


    glfwSwapInterval(1);

    glfwSetKeyCallback(window, key_callback);

    // get version info
    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    printf ("Renderer: %s\n", renderer);
    printf ("OpenGL version supported %s\n", version);


    Scene scene ; 

    while (!glfwWindowShouldClose(window))
    {
        _update_fps_counter (window);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, width, height);

        scene.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (GLFW_PRESS == glfwGetKey (window, GLFW_KEY_ESCAPE)) {
             printf("escape\n");
             glfwSetWindowShouldClose (window, 1);
        }

    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

