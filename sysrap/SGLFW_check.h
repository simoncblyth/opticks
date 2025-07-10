#pragma once
/**
SGLFW_check.h
================

**/

#include "ssys.h"

struct SGLFW_check
{
    static constexpr const char* _level = "SGLFW_check__level" ;
    static int level ;
};

int SGLFW_check::level = ssys::getenvint(_level,0);


inline void SGLFW__check(const char* path, int line, const char* ctx=nullptr, int id=-99, const char* act=nullptr ) // static
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

    const char* delim = ok ? "" : "\n" ;

    if(!ok || SGLFW_check::level  > 0) std::cerr
         << "SGLFW__check " << delim
         //<< "(vi path +line) : "
         << "( vi " << std::setw(55) << path << " " << std::showpos << std::setw(4) << line << std::noshowpos << " ) " << delim
         << " ctx " << std::setw(20) << ( ctx ? ctx : "-" ) << delim
         << " id " << std::setw(4) << id << delim
         << " act " << std::setw(30) << ( act ? act : "-" ) << delim
         << " err "  << std::hex << err << std::dec << delim
         << " errstr " << " : " << ( s ? s : "-" ) << delim
         << " [" <<  SGLFW_check::_level << "] " << SGLFW_check::level << delim
         << "\n"
         ;

    assert( ok );
    if(!ok) std::raise(SIGINT);
}









