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

#include "NGLM.hpp"
#include "NPY.hpp"

#include "GTransforms.hh"


GTransforms::GTransforms(NPY<float>* buf) 
    :
    m_buffer(buf)
{
}

NPY<float>* GTransforms::getBuffer()
{
    return m_buffer ; 
}


GTransforms* GTransforms::make(unsigned int n)
{
    GTransforms* t = new GTransforms();
    for(unsigned int i=0 ; i < n ; i++) t->add();
    return t ;
}

GTransforms* GTransforms::load(const char* path)
{
    NPY<float>* buf = NPY<float>::load(path);
    GTransforms* t = new GTransforms(buf) ;
    return t ; 
}

void GTransforms::save(const char* path)
{
    if(!m_buffer) return ; 
    m_buffer->save(path);
}


void GTransforms::add(const glm::mat4& mat)
{
    if(m_buffer == NULL) m_buffer = NPY<float>::make(0, 4, 4);
    m_buffer->add( glm::value_ptr(mat), 4*4 );
}

void GTransforms::add()
{
    glm::mat4 identity ; 
    add(identity);
}

glm::mat4 GTransforms::get(unsigned int i)
{
    assert( m_buffer && i < m_buffer->getNumItems() && m_buffer->getNumValues(1) == 16);
    glm::mat4 mat = glm::make_mat4(m_buffer->getValues() + i*16) ;
    return mat ; 
}



