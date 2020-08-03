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

#include <cstring>
#include <GL/glew.h>

#include "BStr.hh"
#include "RendererBase.hh"
#include "Prog.hh"
#include "G.hh"

#include "PLOG.hh"

#include "OGLRAP_BODY.hh"


const plog::Severity RendererBase::LEVEL = PLOG::EnvLevel("RendererBase", "DEBUG") ; 


const char* RendererBase::getShaderTag() const 
{
    return m_shadertag ;
}
const char* RendererBase::getShaderDir() const 
{
    return m_shaderdir ;
}
const char* RendererBase::getInclPath() const 
{
    return m_incl_path ;
}

void RendererBase::setVerbosity(unsigned verbosity)
{
    m_verbosity = verbosity ;
    m_shader->setVerbosity(verbosity);
}

void RendererBase::setNoFrag(bool nofrag)
{   
    m_shader->setNoFrag(nofrag) ; 
}

const char* RendererBase::getName() const  // shadertag if index has not been set 
{
    return m_name ? m_name : m_shadertag  ;    
}
void RendererBase::setIndexBBox(unsigned index, bool bbox)
{
    m_index = index ;   
    m_bbox = bbox ; 
    m_name = BStr::concat<unsigned>(m_shadertag, m_index, m_bbox ? "bb" : "" ); 
}

unsigned RendererBase::getIndex() const
{
    return m_index ; 
}


RendererBase::RendererBase(const char* tag, const char* dir, const char* incl_path, bool ubo)
    :
    m_shader(NULL),
    m_program(-1),
    m_verbosity(0),
    m_shaderdir(dir ? strdup(dir) : getenv("SHADER_DIR")),
    m_shadertag(strdup(tag)),
    m_incl_path(incl_path ? strdup(incl_path) : getenv("SHADER_INCL_PATH")),
    m_index(0),
    m_name(NULL)
{
    // no OpenGL context needed, just reads sources
    // so cannot errcheck here (gives bad access)
    m_shader = new Prog(m_shaderdir, m_shadertag, m_incl_path, ubo ); 
}


void RendererBase::make_shader()
{
    LOG(LEVEL)
         << " shaderdir " << getShaderDir()
         << " shadertag " << getShaderTag()
         ;

    G::ErrCheck("RendererBase::make_shader.[", true);

    m_shader->createAndLink();
    //m_shader->Summary("RendererBase::make_shader");
    m_program = m_shader->getId(); 

    LOG(debug) << "RendererBase::make_shader "
              << " shaderdir " << getShaderDir()
              << " shadertag " << getShaderTag()
              << " program " <<  m_program ;

    G::ErrCheck("RendererBase::make_shader.]", true);
}


void RendererBase::create_shader()
{
    G::ErrCheck("RendererBase::create_shader.[", true);
    m_shader->createOnly();
    m_program = m_shader->getId(); 
    G::ErrCheck("RendererBase::create_shader.]", true);
}

void RendererBase::link_shader()
{
    G::ErrCheck("RendererBase::link_shader.[", true);
    m_shader->linkAndValidate();
    G::ErrCheck("RendererBase::link_shader.]", true);
}




