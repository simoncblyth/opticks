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


#include <cstdio>
#include <cstring>
#include <cassert>

#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

#include "NPY_FLAGS.hh"

MultiViewNPY::MultiViewNPY(const char* name)
    :   
    m_name(strdup(name))
{
}

MultiViewNPY::~MultiViewNPY()
{
    free((char*)m_name); 
}

const char* MultiViewNPY::getName()
{
    return m_name ;
}


void MultiViewNPY::add(ViewNPY* vec)
{ 
    if(m_vecs.size() > 0)
    {
        ViewNPY* prior = m_vecs.back();
        assert(prior->getNPY() == vec->getNPY() && "LIMITATION : all ViewNPY in a MultiViewNPY must be views of the same underlying NPY");
    }
    m_vecs.push_back(vec);
    vec->setParent(this);
}

unsigned int  MultiViewNPY::getNumVecs()
{ 
    return m_vecs.size();
}

ViewNPY* MultiViewNPY::operator [](const char* name)
{
    return find(name);
}


ViewNPY* MultiViewNPY::operator [](unsigned int index)
{
    return index < m_vecs.size() ? m_vecs[index] : NULL ;
}


ViewNPY* MultiViewNPY::find(const char* name)
{
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        ViewNPY* vnpy = m_vecs[i];
        if(strcmp(name, vnpy->getName())==0) return vnpy ;
    }
    return NULL ; 
}

void MultiViewNPY::Print(const char* msg)
{
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        ViewNPY* vnpy = m_vecs[i];
        vnpy->Print(msg);
    }
}

void MultiViewNPY::Summary(const char* msg)
{
    printf("[%s]\n", msg);
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        ViewNPY* vnpy = m_vecs[i];
        vnpy->Summary(msg);
    }
}
