#pragma once

#include <cassert>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#ifndef GLFW_TRUE
#define GLFW_TRUE true
#endif


struct SGLFW
{
    static void check(const char* path, int line); 
    static void print_shader_info_log(unsigned id); 
    static void error_callback(int error, const char* description); 
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods); 
}; 




inline void SGLFW::check(const char* path, int line) 
{
    GLenum err = glGetError() ;   
    bool ok = err == GL_NO_ERROR ;
    const char* s = NULL ; 
    switch(err)
    {   
        case GL_INVALID_ENUM:      s = "GL_INVALID_ENUM" ; break ; 
        case GL_INVALID_VALUE:     s = "GL_INVALID_VALUE" ; break ; 
        case GL_INVALID_OPERATION: s = "GL_INVALID_OPERATION" ; break ; 
        case GL_STACK_OVERFLOW:    s = "GL_STACK_OVERFLOW" ; break ;   
        case GL_STACK_UNDERFLOW:   s = "GL_STACK_UNDERFLOW" ; break ;   
        case GL_OUT_OF_MEMORY:     s = "GL_OUT_OF_MEMORY" ; break ;   
        case GL_CONTEXT_LOST:      s = "GL_CONTEXT_LOST" ; break ;
        case GL_INVALID_FRAMEBUFFER_OPERATION: s = "GL_INVALID_FRAMEBUFFER_OPERATION" ; break ;
    }   
    if(!ok) std::cout << "SGLFW::check OpenGL ERROR " << path << " : " << line << " : " << std::hex << err << std::dec << " : " << s << std::endl ; 
    assert( ok );  
}


inline void SGLFW::print_shader_info_log(unsigned id) 
{
    int max_length = 2048;
    int actual_length = 0;
    char log[2048];

    glGetShaderInfoLog(id, max_length, &actual_length, log);
    SGLFW::check(__FILE__, __LINE__ );  

    printf ("shader info log for GL index %u:\n%s\n", id, log);
    assert(0);
}
inline void SGLFW::error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

/**

some ideas on key handling :  UseOpticksGLFW/UseOpticksGLFW.cc 

https://stackoverflow.com/questions/55573238/how-do-i-do-a-proper-input-class-in-glfw-for-a-game-engine

https://learnopengl.com/Getting-started/Camera

THIS NEED TO TALK TO SGLM::INSTANCE changing viewpoint 

**/
inline void SGLFW::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {   
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }   
}



