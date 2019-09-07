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

#include "G.hh"
#include "RContext.hh"
#include "PLOG.hh"


const plog::Severity RContext::LEVEL = PLOG::EnvLevel("RContext", "DEBUG") ; 



// gl/InstLODCullContext.h
struct InstLODCullContext
{
    glm::mat4 ModelViewProjection ;
    glm::mat4 ModelView ;
    glm::vec4 LODCUT ;
};


const char* RContext::uniformBlockName = "InstLODCullContext" ;


RContext::RContext()
    :
    uniform(new InstLODCullContext)
{
}


void RContext::init()
{
    initUniformBuffer();
}

void RContext::initUniformBuffer()
{
    LOG(info) << "RContext::initUniformBuffer" ; 

     // same UBO can be used from all shaders
    glGenBuffers(1, &this->uniformBO);
    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(InstLODCullContext), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, this->uniformBO);

    G::ErrCheck("RContext::initUniformBuffer", true);
}

void RContext::bindUniformBlock(GLuint program)
{
    GLuint uniformBlockIndex = glGetUniformBlockIndex(program,  uniformBlockName ) ;
    assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform otherwise it gets optimized away") ;
    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(program, uniformBlockIndex,  uniformBlockBinding );

    G::ErrCheck("RContext::initUniformBuffer", true);
}

void RContext::update( const glm::mat4& world2clip, const glm::mat4& world2eye, const glm::vec4& lodcut)
{
    //LOG(info) << "RContext::update" ; 

    uniform->ModelViewProjection = world2clip  ;  
    uniform->ModelView = world2eye  ;  
    uniform->LODCUT = lodcut  ;  

    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(InstLODCullContext), this->uniform);
}


