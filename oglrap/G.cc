#include <sstream>

#include <GL/glew.h>
#include "G.hh"
#include "PLOG.hh"

const char* G::GL_INVALID_ENUM_ = "GL_INVALID_ENUM" ; 
const char* G::GL_INVALID_VALUE_ = "GL_INVALID_VALUE" ; 
const char* G::GL_INVALID_OPERATION_ = "GL_INVALID_OPERATION" ; 
const char* G::GL_STACK_OVERFLOW_ = "GL_STACK_OVERFLOW" ; 
const char* G::GL_STACK_UNDERFLOW_ = "GL_STACK_UNDERFLOW" ; 
const char* G::GL_OUT_OF_MEMORY_ = "GL_OUT_OF_MEMORY" ; 
const char* G::GL_INVALID_FRAMEBUFFER_OPERATION_ = "GL_INVALID_FRAMEBUFFER_OPERATION" ; 
const char* G::GL_CONTEXT_LOST_ = "GL_CONTEXT_LOST" ;
const char* G::OTHER_ = "OTHER?" ;


const char* G::GL_VERTEX_SHADER_ = "GL_VERTEX_SHADER" ; 
const char* G::GL_GEOMETRY_SHADER_ = "GL_GEOMETRY_SHADER" ; 
const char* G::GL_FRAGMENT_SHADER_ = "GL_FRAGMENT_SHADER" ; 

const char* G::Shader( GLenum type )
{
    const char* s = OTHER_ ; 
    switch(type)
    {
       case GL_VERTEX_SHADER: s = GL_VERTEX_SHADER_ ; break ; 
       case GL_FRAGMENT_SHADER: s = GL_FRAGMENT_SHADER_ ; break ; 
       case GL_GEOMETRY_SHADER: s = GL_GEOMETRY_SHADER_ ; break ; 
    }
    return s ; 
}


const char* G::Err( GLenum err )
{
    const char* s = OTHER_ ; 
    switch(err)
    {
        case GL_INVALID_ENUM: s = GL_INVALID_ENUM_ ; break ; 
        case GL_INVALID_VALUE: s = GL_INVALID_VALUE_ ; break ; 
        case GL_INVALID_OPERATION: s = GL_INVALID_OPERATION_ ; break ; 
        case GL_STACK_OVERFLOW : s = GL_STACK_OVERFLOW_ ; break ;  
        case GL_STACK_UNDERFLOW : s = GL_STACK_UNDERFLOW_ ; break ;  
        case GL_OUT_OF_MEMORY : s = GL_OUT_OF_MEMORY_ ; break ;  
        case GL_INVALID_FRAMEBUFFER_OPERATION : s = GL_INVALID_FRAMEBUFFER_OPERATION_ ; break ;
        case GL_CONTEXT_LOST : s = GL_CONTEXT_LOST_ ; break ;
    }
    return s ; 
}


std::string G::ErrCheck(const char* msg, bool harikari)
{
    std::stringstream ss ; 

    GLenum err ;
    if ((err = glGetError()) != GL_NO_ERROR)
    {          
          ss
            << "G::ErrCheck " 
            << msg 
            << " WARNING : OpenGL error code: "
            << std::hex << err << std::dec
            << " err " << Err(err) 
            ;
    }   

    std::string err_ = ss.str();

    std::string empty ; 
    if(err == GL_INVALID_ENUM ) 
    {
         LOG(warning) << "G::ErrCheck ignoring " << err_  ; 
         return empty ; 
    } 

    if(!err_.empty() && harikari)
    {
        LOG(fatal) << err_ ; 
        assert(0);
    } 

    return err_ ;
}

