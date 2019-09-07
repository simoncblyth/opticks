/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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



