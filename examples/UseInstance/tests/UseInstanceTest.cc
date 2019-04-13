/*
Fairly Minimal Example of OpenGL Instancing 
===============================================

Using glVertexAttribDivisor, glDrawArraysInstanced
    
*/

#include <vector>
#include <iostream>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"
#include "Renderer.hh"

const char* vertSrc = R"glsl(

    #version 400 core
    layout(location = 0) in vec4 vPosition ;
    layout(location = 1) in vec4 iPosition ;
    void main()
    {
        //gl_Position = vec4( vPosition.x, vPosition.y, vPosition.z, 1.0 ) ;  
        gl_Position = vec4( vPosition.x + iPosition.x, vPosition.y + iPosition.y, vPosition.z + iPosition.z, 1.0 ) ;  
    }
)glsl";


const char* fragSrc = R"glsl(
    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";

struct V { float x,y,z,w ; };
static const unsigned NUM_VPOS = 3 ; 

V vpos[NUM_VPOS] = 
{
    { -0.1f , -0.1f,  0.f,  1.f }, 
    { -0.1f ,  0.1f,  0.f,  1.f },
    {  0.f ,   0.f,   0.f,  1.f }
};

static const unsigned NUM_IPOS = 8 ; 
V ipos[NUM_IPOS] = 
{
    {   0.1f ,   0.1f,   0.f,  1.f }, 
    {   0.2f ,   0.2f,   0.f,  1.f },
    {   0.3f ,   0.3f,   0.f,  1.f },
    {   0.4f ,   0.4f,   0.f,  1.f },
    {  -0.1f ,  -0.1f,   0.f,  1.f }, 
    {  -0.2f ,  -0.2f,   0.f,  1.f },
    {  -0.3f ,  -0.3f,   0.f,  1.f },
    {  -0.4f ,  -0.4f,   0.f,  1.f }
};


int main()
{
    Frame frame ; 
    Prog prog(vertSrc, NULL, fragSrc ) ; 
    prog.compile();
    prog.create();
    prog.link();

    Buf v( sizeof(vpos),vpos ) ; 
    Buf i( sizeof(ipos),ipos ) ; 

    Renderer rdr ; 
    rdr.upload(&v);
    rdr.upload(&i);

    GLint vPosition = prog.getAttribLocation("vPosition");
    glBindBuffer(GL_ARRAY_BUFFER, v.id);
    glEnableVertexAttribArray(vPosition);
    {
        GLuint index = vPosition ; 
        GLint  size = 4 ;         // Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, 4.
        GLenum type = GL_FLOAT ;
        GLboolean normalized = GL_FALSE ;
        GLsizei stride = 4*sizeof(float) ;  // byte offset between consecutive generic vertex attributes
        const GLvoid* offset = NULL ;

        glVertexAttribPointer(index, size, type, normalized, stride, offset);
    }

    GLint iPosition = prog.getAttribLocation("iPosition");
    glBindBuffer(GL_ARRAY_BUFFER, i.id);
    glEnableVertexAttribArray(iPosition);
    {
        GLuint index = iPosition ; 
        GLint  size = 4 ;         // Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, 4.
        GLenum type = GL_FLOAT ;
        GLboolean normalized = GL_FALSE ;
        GLsizei stride = 4*sizeof(float) ;  // byte offset between consecutive generic vertex attributes
        const GLvoid* offset = NULL ;

        glVertexAttribPointer(index, size, type, normalized, stride, offset);
        glVertexAttribDivisor(index, 1 );
    }

    while (!glfwWindowShouldClose(frame.window))
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        {
            GLenum mode = GL_TRIANGLES ;
            GLint first = 0 ; 
            GLsizei count = NUM_VPOS ;  // number of indices to render, NB not number of prim
            GLsizei primCount = NUM_IPOS ;  
            glDrawArraysInstanced(mode, first, count, primCount );
            //glDrawArrays(mode, first, count );
        }

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    prog.destroy();
    rdr.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


