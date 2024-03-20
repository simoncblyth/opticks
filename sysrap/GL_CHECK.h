#pragma once
/**
GL_CHECK.h
============

Adapted from SDK/sutil/Exception.h 

**/

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#define DO_GL_CHECK
#ifdef DO_GL_CHECK

#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
           ::glCheck( #call, __FILE__, __LINE__ );                         \
    } while( false )


#define GL_CHECK_ERRORS() ::glCheckErrors( __FILE__, __LINE__ )

#else
#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
    } while( 0 )
#define GL_CHECK_ERRORS()                                                      \
    do                                                                         \
    {                                                                          \
        ;                                                                      \
    } while( 0 )
#endif

inline const char* getGLErrorString( GLenum error )
{
    switch( error )
    {
        case GL_NO_ERROR:          return "No error";
        case GL_INVALID_ENUM:      return "Invalid enum";
        case GL_INVALID_VALUE:     return "Invalid value";
        case GL_INVALID_OPERATION: return "Invalid operation";
        //case GL_STACK_OVERFLOW:  return "Stack overflow";
        //case GL_STACK_UNDERFLOW: return "Stack underflow";
        case GL_OUT_OF_MEMORY:     return "Out of memory";
        //case GL_TABLE_TOO_LARGE: return "Table too large";
        default:                   return "Unknown GL error";
    }
}

inline void glCheck( const char* call, const char* file, unsigned int line )
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString( err ) << " at " << file << "("
           << line << "): " << call << '\n';
        std::cerr << ss.str() << std::endl;
        throw std::runtime_error( ss.str().c_str() );
    }
}

inline void glCheckErrors( const char* file, unsigned int line )
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString( err ) << " at " << file << "("
           << line << ")";
        std::cerr << ss.str() << std::endl;
        throw std::runtime_error( ss.str().c_str() );
    }
}

inline void checkGLError()
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::ostringstream oss;
        do
        {
            oss << "GL error: " << getGLErrorString( err ) << '\n';
            err = glGetError();
        } while( err != GL_NO_ERROR );

        throw std::runtime_error( oss.str().c_str() );
    }
}

