#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#ifndef GLFW_TRUE
#define GLFW_TRUE true
#endif

struct SGLFW_Type
{
    static constexpr const char* GL_BYTE_           = "GL_BYTE" ; 
    static constexpr const char* GL_UNSIGNED_BYTE_  = "GL_UNSIGNED_BYTE" ; 
    static constexpr const char* GL_SHORT_          = "GL_SHORT" ; 
    static constexpr const char* GL_UNSIGNED_SHORT_ = "GL_UNSIGNED_SHORT" ; 
    static constexpr const char* GL_INT_            = "GL_INT" ; 
    static constexpr const char* GL_UNSIGNED_INT_   = "GL_UNSIGNED_INT" ; 
    static constexpr const char* GL_HALF_FLOAT_     = "GL_HALF_FLOAT" ; 
    static constexpr const char* GL_FLOAT_          = "GL_FLOAT" ; 
    static constexpr const char* GL_DOUBLE_         = "GL_DOUBLE" ; 

    static const char* Name(GLenum type); 
    static GLenum      Type(const char* name); 
};

inline const char* SGLFW_Type::Name(GLenum type)
{
    const char* s = nullptr ; 
    switch(type)
    {
        case GL_BYTE:           s = GL_BYTE_           ; break ; 
        case GL_UNSIGNED_BYTE:  s = GL_UNSIGNED_BYTE_  ; break ; 
        case GL_SHORT:          s = GL_SHORT_          ; break ; 
        case GL_UNSIGNED_SHORT: s = GL_UNSIGNED_SHORT_ ; break ; 
        case GL_INT:            s = GL_INT_            ; break ; 
        case GL_UNSIGNED_INT:   s = GL_UNSIGNED_INT_   ; break ; 
        case GL_HALF_FLOAT:     s = GL_HALF_FLOAT_     ; break ;
        case GL_FLOAT:          s = GL_FLOAT_          ; break ;
        case GL_DOUBLE:         s = GL_DOUBLE_         ; break ;
        default:                s = nullptr            ; break ;
    }
    return s ; 
}

inline GLenum SGLFW_Type::Type(const char* name)
{
    GLenum type = 0 ; 
    if( strcmp( name, GL_BYTE_) == 0 )           type = GL_BYTE ; 
    if( strcmp( name, GL_UNSIGNED_BYTE_) == 0 )  type = GL_UNSIGNED_BYTE ; 
    if( strcmp( name, GL_SHORT_) == 0 )          type = GL_SHORT ; 
    if( strcmp( name, GL_UNSIGNED_SHORT_) == 0 ) type = GL_UNSIGNED_SHORT ; 
    if( strcmp( name, GL_INT_) == 0 )            type = GL_INT ; 
    if( strcmp( name, GL_UNSIGNED_INT_) == 0 )   type = GL_UNSIGNED_INT ; 
    if( strcmp( name, GL_HALF_FLOAT_) == 0 )     type = GL_HALF_FLOAT ; 
    if( strcmp( name, GL_FLOAT_) == 0 )          type = GL_FLOAT ; 
    if( strcmp( name, GL_DOUBLE_) == 0 )         type = GL_DOUBLE ; 
    return type ; 
}


struct SGLFW_Attribute
{
    const char* name_spec ; 
    char* name ; 
    const char* spec ; 
    std::vector<std::string> elem ; 

    GLuint index ; 
    GLint size ; 
    GLenum type ; 
    GLboolean normalized ; 
    GLsizei stride ; 
    const void* pointer ; 

    bool     iatt ;      
    unsigned offset ; 

    SGLFW_Attribute( const char* name_spec ); 
    std::string desc() const ;  
};


SGLFW_Attribute::SGLFW_Attribute(const char* name_spec_)
    :
    name_spec(strdup(name_spec_)),
    name(strdup(name_spec_)),
    spec(nullptr), 
    index(0),
    size(0),
    type(0),
    normalized(false),
    stride(0),
    pointer(nullptr),
    iatt(false),
    offset(0)
{
    char* p = strchr(name, ':' );
    assert(p); 
    *p = '\0' ; 
    spec = p+1 ; 
    
    char delim = ',' ; 
    std::stringstream ss; 
    ss.str(spec)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 
}


std::string SGLFW_Attribute::desc() const 
{
    std::stringstream ss ; 
    ss << "SGLFW_Attribute::desc" << std::endl 
       << "name_spec["  << name_spec << "]" << std::endl 
       << "name["  << name << "]" << std::endl 
       << "spec["  << spec << "]" << std::endl 
       << "index:" << index << std::endl 
       << "size:" << size << std::endl
       << "type:" << type << std::endl
       << "normalized:" << normalized << std::endl
       << "stride:" << stride << std::endl
       << "pointer:" << pointer << std::endl
       << "iatt:" << iatt << std::endl
       << "offset:" << offset << std::endl
       << "elem.size:" << elem.size() << std::endl 
       ;

    for(unsigned i=0 ; i < elem.size() ; i++ ) ss << "elem[" << i << "]:" << elem[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}




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



