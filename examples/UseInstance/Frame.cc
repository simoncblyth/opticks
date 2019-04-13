#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cassert>


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Frame.hh"


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

Frame::Frame()  
  :
  window(NULL)
{
   init();
} 

void Frame::init()
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

#if defined __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3); 
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2); 
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // this incantation gives
    //    Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    //    OpenGL version supported 4.1 NVIDIA-10.33.0 387.10.10.10.40.105

#elif defined _MSC_VER
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4); 
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1); 
 
#elif __linux
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4); 
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1); 

    //  executing UseInstanceTest
    // Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    // Frame::gl_init_window OpenGL version supported 4.1.0 NVIDIA 418.56
#endif


    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);


    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();


   // get version info
    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    std::cout << "Frame::gl_init_window Renderer: " << renderer << std::endl ; 
    std::cout << "Frame::gl_init_window OpenGL version supported " <<  version << std::endl ;
}


void Frame::destroy()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}


