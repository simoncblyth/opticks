// /Users/blyth/env/graphics/glfw/glfwminimal/glfwminimal.cc
// http://www.glfw.org/docs/latest/quick.html

#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>


#include <string>
#include <sstream>
#include <iostream>

#include "glfw_keyname.h"



enum { NUM_KEYS = 512  } ;
static bool keys_down[NUM_KEYS] ;


enum { 
      e_shift   = 1 << 0,  
      e_control = 1 << 1,  
      e_option  = 1 << 2,  
      e_command = 1 << 3 
    } ; 


static int getModifiers()
{
    unsigned modifiers = 0 ; 
    if( keys_down[GLFW_KEY_LEFT_SHIFT]   || keys_down[GLFW_KEY_RIGHT_SHIFT] )    modifiers |= e_shift ;
    if( keys_down[GLFW_KEY_LEFT_CONTROL] || keys_down[GLFW_KEY_RIGHT_CONTROL] )  modifiers |= e_control ;
    if( keys_down[GLFW_KEY_LEFT_ALT]     || keys_down[GLFW_KEY_RIGHT_ALT] )      modifiers |= e_option ;
    if( keys_down[GLFW_KEY_LEFT_SUPER]   || keys_down[GLFW_KEY_RIGHT_SUPER] )    modifiers |= e_command ;
    return modifiers ; 
}

std::string descModifiers(int modifiers)
{
    std::stringstream ss ;  
    if( modifiers & e_shift ) ss << "shift " ; 
    if( modifiers & e_control ) ss << "control " ; 
    if( modifiers & e_option ) ss << "option " ; 
    if( modifiers & e_command ) ss << "command " ; 
    return ss.str();  
}


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if( action == GLFW_PRESS )
    {
        keys_down[key] = true ;   

        const char* keyname = glfw_keyname(key) ; 
        std::cout << " pressed : " << keyname << std::endl ; 

    }
    else if (action == GLFW_RELEASE )
    {
        keys_down[key] = false ;   
    }


    int modifiers = getModifiers(); 
    if(modifiers != 0)
    std::cout << descModifiers(modifiers) << std::endl ; 

}


int main(void)
{
    GLFWwindow* window;

    for(unsigned i=0 ; i < NUM_KEYS ; i++) keys_down[i] = false ;

    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);
    window = glfwCreateWindow(640, 480, "UseOpticksGLFW : minimal usage of OpenGL via GLFW : press ESCAPE to exit", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

    int count(0);
    bool exitloop(false);
    int renderlooplimit(200); 

    while (!glfwWindowShouldClose(window) && !exitloop)
    {
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
        glBegin(GL_TRIANGLES);
        glColor3f(1.f, 0.f, 0.f);
        glVertex3f(-0.6f, -0.4f, 0.f);
        glColor3f(0.f, 1.f, 0.f);
        glVertex3f(0.6f, -0.4f, 0.f);
        glColor3f(0.f, 0.f, 1.f);
        glVertex3f(0.f, 0.6f, 0.f);
        glEnd();
        glfwSwapBuffers(window);
        glfwPollEvents();

        count++ ; 
        //std::cout << count << std::endl ;  

        exitloop = renderlooplimit > 0 && count > renderlooplimit ;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

