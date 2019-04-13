
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Renderer.hh"
#include "Buf.hh"


GLuint _upload(GLenum target, unsigned num_bytes, void* ptr, GLenum usage )
{
    GLuint buffer_id ;
    glGenBuffers(1, &buffer_id);
    glBindBuffer(target, buffer_id);
    glBufferData(target, num_bytes, ptr, usage);
    return buffer_id ;
}

Renderer::Renderer()
{
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao); // no target argument, because there is only one target for VAO 
   /*
    https://www.khronos.org/opengl/wiki/Vertex_Specification#Vertex_Array_Object
   */
}

void Renderer::upload(Buf* buf)
{
    buf->id = _upload( GL_ARRAY_BUFFER, buf->num_bytes, buf->ptr, GL_STATIC_DRAW);
    buffers.push_back(buf);
}

void Renderer::destroy()
{
    for(unsigned i=0 ; i < buffers.size() ; i++)
    {
        Buf* buf = buffers[i]; 
        const GLuint id = buf->id ; 
        glDeleteBuffers(1, &id);
    }
    glDeleteVertexArrays(1, &vao);
}



