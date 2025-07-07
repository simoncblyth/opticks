#pragma once
/**
SGLFW_check.h
================

**/

inline void SGLFW__check(const char* path, int line, const char* ctx=nullptr, int id=-99 ) // static
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
#ifdef GL_VERSION_4_5
        case GL_CONTEXT_LOST:      s = "GL_CONTEXT_LOST" ; break ;
#endif
        case GL_INVALID_FRAMEBUFFER_OPERATION: s = "GL_INVALID_FRAMEBUFFER_OPERATION" ; break ;
    }
    if(!ok) std::cerr
         << "SGLFW__check OpenGL ERROR " << "\n"
         << "(vi path +line) : ( vi " << path << " +" << line << " ) " << "\n"
         << " ctx " << ( ctx ? ctx : "-" ) << "\n"
         << " id " << id << "\n"
         << " err "  << std::hex << err << std::dec << "\n"
         << " errstr " << " : " << ( s ? s : "-" ) << "\n"
         ;

    assert( ok );
    if(!ok) std::raise(SIGINT);
}









