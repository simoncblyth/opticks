#pragma once

/**
SGLFW_Buffer : minimal OpenGL buffer wrapper
---------------------------------------------

Old Opticks oglrap handled multi-buffers using RBuf held by Renderer
See::

   Renderer::createVertexArray

**/

struct SGLFW_Buffer
{
    const char* name ; 
    int num_bytes ;
    const void* data ;
    GLenum target ;
    GLenum usage ;
    GLuint id ;

    SGLFW_Buffer( const char* name, int num_bytes, const void* data, GLenum target, GLenum usage  );

    void bind();
    void upload();
    void unbind();
};

inline SGLFW_Buffer::SGLFW_Buffer( const char* _name, int num_bytes_, const void* data_ , GLenum target_, GLenum usage_ )
    :
    name( _name ? strdup(_name) : nullptr ),
    num_bytes(num_bytes_),
    data(data_),
    target(target_),
    usage(usage_),
    id(0)
{
    glGenBuffers(1, &id );                         SGLFW__check(__FILE__, __LINE__, name, id);
}

inline void SGLFW_Buffer::bind()
{
    glBindBuffer(target, id);                      SGLFW__check(__FILE__, __LINE__, name, id);
}

inline void SGLFW_Buffer::upload()
{
    glBufferData(target, num_bytes, data, usage ); SGLFW__check(__FILE__, __LINE__, name, id);
}

inline void SGLFW_Buffer::unbind()
{
    glBindBuffer(target, 0);                      SGLFW__check(__FILE__, __LINE__, name, id);
}


