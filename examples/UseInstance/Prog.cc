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

#include "Prog.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

Prog::Prog(const char* vertSrc_, const char* geomSrc_,  const char* fragSrc_)
        :
        vertSrc(vertSrc_),
        geomSrc(geomSrc_),
        fragSrc(fragSrc_),
        vert(vertSrc != NULL),
        geom(geomSrc != NULL),
        frag(fragSrc != NULL)
{
}

void Prog::compile()
{
    if(vert)
    {
        vertShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertShader, 1, &vertSrc, nullptr);
        glCompileShader(vertShader);
    }
    if(geom)
    {
        geomShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geomShader, 1, &geomSrc, nullptr);
        glCompileShader(geomShader); 
    }
    if(frag)
    {
        fragShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragShader, 1, &fragSrc, nullptr);
        glCompileShader(fragShader); 
    }

}

void Prog::create()
{
    program = glCreateProgram();
    if(vert) glAttachShader(program, vertShader);
    if(geom) glAttachShader(program, geomShader);
    if(frag) glAttachShader(program, fragShader);
}

void Prog::link()
{
    glLinkProgram(program);
    glUseProgram(program);
} 

void Prog::destroy()
{
    glDeleteProgram(program);
    if(geom) glDeleteShader(geomShader);
    if(vert) glDeleteShader(vertShader);
    if(frag) glDeleteShader(fragShader);
}

int Prog::getAttribLocation(const char* att)
{
    return glGetAttribLocation(program, att );
}




