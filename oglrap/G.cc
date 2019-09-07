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

#include <sstream>

#include <GL/glew.h>
#include "G.hh"
#include "PLOG.hh"

const char* G::GL_INVALID_ENUM_ = "GL_INVALID_ENUM" ; 
const char* G::GL_INVALID_VALUE_ = "GL_INVALID_VALUE" ; 
const char* G::GL_INVALID_OPERATION_ = "GL_INVALID_OPERATION" ; 
const char* G::GL_STACK_OVERFLOW_ = "GL_STACK_OVERFLOW" ; 
const char* G::GL_STACK_UNDERFLOW_ = "GL_STACK_UNDERFLOW" ; 
const char* G::GL_OUT_OF_MEMORY_ = "GL_OUT_OF_MEMORY" ; 
const char* G::GL_INVALID_FRAMEBUFFER_OPERATION_ = "GL_INVALID_FRAMEBUFFER_OPERATION" ; 
const char* G::GL_CONTEXT_LOST_ = "GL_CONTEXT_LOST" ;
const char* G::OTHER_ = "OTHER?" ;


const char* G::GL_VERTEX_SHADER_ = "GL_VERTEX_SHADER" ; 
const char* G::GL_GEOMETRY_SHADER_ = "GL_GEOMETRY_SHADER" ; 
const char* G::GL_FRAGMENT_SHADER_ = "GL_FRAGMENT_SHADER" ; 

bool G::VERBOSE = false ; 

const char* G::Shader( GLenum type )
{
    const char* s = OTHER_ ; 
    switch(type)
    {
       case GL_VERTEX_SHADER: s = GL_VERTEX_SHADER_ ; break ; 
       case GL_FRAGMENT_SHADER: s = GL_FRAGMENT_SHADER_ ; break ; 
       case GL_GEOMETRY_SHADER: s = GL_GEOMETRY_SHADER_ ; break ; 
    }
    return s ; 
}


const char* G::Err( GLenum err )
{
    const char* s = OTHER_ ; 
    switch(err)
    {
        case GL_INVALID_ENUM: s = GL_INVALID_ENUM_ ; break ; 
        case GL_INVALID_VALUE: s = GL_INVALID_VALUE_ ; break ; 
        case GL_INVALID_OPERATION: s = GL_INVALID_OPERATION_ ; break ; 
        case GL_STACK_OVERFLOW : s = GL_STACK_OVERFLOW_ ; break ;  
        case GL_STACK_UNDERFLOW : s = GL_STACK_UNDERFLOW_ ; break ;  
        case GL_OUT_OF_MEMORY : s = GL_OUT_OF_MEMORY_ ; break ;  
        case GL_INVALID_FRAMEBUFFER_OPERATION : s = GL_INVALID_FRAMEBUFFER_OPERATION_ ; break ;
        case GL_CONTEXT_LOST : s = GL_CONTEXT_LOST_ ; break ;
    }
    return s ; 
}


bool G::ErrCheck(const char* msg, bool harikari )
{
    if(G::VERBOSE) LOG(fatal) << msg ; 
    GLenum err = glGetError()  ;
    bool ok = err == GL_NO_ERROR ; 
    if (!ok)
    {          
        LOG(fatal)
            << "G::ErrCheck " 
            << msg 
            << " : "
            << std::hex << err << std::dec
            << " err " << Err(err) 
            ;

        if(harikari) assert(0); 
    }   
    return ok ;
}

