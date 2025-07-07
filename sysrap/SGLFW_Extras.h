#pragma once
/**
SGLFW_Extras.h : Toggle, GLboolean, bool, GLenum, Attrib, Buffer, VAO
======================================================================

SGLFW__check


SGLFW_Buffer
   minimal OpenGL buffer wrapper

SGLFW_VAO
   minimal Vertex Array wrapper


**/

inline void SGLFW__check(const char* path, int line, const char* ctx=nullptr) // static
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
         << " err "  << std::hex << err << std::dec << "\n"
         << " errstr " << " : " << ( s ? s : "-" ) << "\n"
         ;

    assert( ok );
    if(!ok) std::raise(SIGINT);
}







/**
SGLFW_Buffer : minimal OpenGL buffer wrapper
---------------------------------------------

Old Opticks oglrap handled multi-buffers using RBuf held by Renderer
See::

   Renderer::createVertexArray

**/

struct SGLFW_Buffer
{
    int num_bytes ;
    const void* data ;
    GLenum target ;
    GLenum usage ;
    GLuint id ;

    SGLFW_Buffer( int num_bytes, const void* data, GLenum target, GLenum usage  );

    void bind();
    void upload();
    void unbind();
};

inline SGLFW_Buffer::SGLFW_Buffer( int num_bytes_, const void* data_ , GLenum target_, GLenum usage_ )
    :
    num_bytes(num_bytes_),
    data(data_),
    target(target_),
    usage(usage_),
    id(0)
{
    glGenBuffers(1, &id );                         SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_Buffer::bind()
{
    glBindBuffer(target, id);                      SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_Buffer::upload()
{
    glBufferData(target, num_bytes, data, usage ); SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_Buffer::unbind()
{
    glBindBuffer(target, 0);                      SGLFW__check(__FILE__, __LINE__);
}



/**
SGLFW_VAO : Minimal Vertex Array wrapper
--------------------------------------------
**/

struct SGLFW_VAO
{
    GLuint id ;

    SGLFW_VAO();
    void init();
    void bind();
    void unbind();
};

inline SGLFW_VAO::SGLFW_VAO()
    :
    id(-1)
{
    init();
}

inline void SGLFW_VAO::init()
{
    //printf("SGLFW_VAO::init\n");
    glGenVertexArrays (1, &id);  SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_VAO::bind()
{
    glBindVertexArray(id);        SGLFW__check(__FILE__, __LINE__);
}
inline void SGLFW_VAO::unbind()
{
    glBindVertexArray(0);        SGLFW__check(__FILE__, __LINE__);
}



