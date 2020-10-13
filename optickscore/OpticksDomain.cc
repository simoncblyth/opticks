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
#include "GLMFormat.hpp"
#include "OpticksDomain.hh"

#include "PLOG.hh"



OpticksDomain::OpticksDomain()
    :
    m_fdom(NULL),
    m_idom(NULL),
    m_space_domain(0.f,0.f,0.f,0.f),
    m_time_domain(0.f,0.f,0.f,0.f),
    m_wavelength_domain(0.f,0.f,0.f,0.f),
    m_settings(0,0,0,0)
{
    init();
}

void OpticksDomain::init()
{
}


void OpticksDomain::setFDomain(NPY<float>* fdom)
{
    m_fdom = fdom ; 
}
void OpticksDomain::setIDomain(NPY<int>* idom)
{
    m_idom = idom ; 
}

NPY<float>* OpticksDomain::getFDomain() const 
{
    return m_fdom ; 
}
NPY<int>* OpticksDomain::getIDomain() const 
{
    return m_idom ; 
}



void OpticksDomain::setSpaceDomain(const glm::vec4& space_domain)
{
    m_space_domain = space_domain ; 
}
void OpticksDomain::setTimeDomain(const glm::vec4& time_domain)
{
    m_time_domain = time_domain  ; 
}
void OpticksDomain::setWavelengthDomain(const glm::vec4& wavelength_domain)
{
    m_wavelength_domain = wavelength_domain  ; 
}


const glm::vec4& OpticksDomain::getSpaceDomain() const
{
    return m_space_domain ; 
}
const glm::vec4& OpticksDomain::getTimeDomain() const
{
    return m_time_domain ;
}
const glm::vec4& OpticksDomain::getWavelengthDomain() const 
{ 
    return m_wavelength_domain ; 
}





void OpticksDomain::updateBuffer()
{
    NPY<float>* fdom = getFDomain();
    if(fdom)
    {
        fdom->setQuad(m_space_domain     , 0);
        fdom->setQuad(m_time_domain      , 1);
        fdom->setQuad(m_wavelength_domain, 2);
    }
    else
    {
        LOG(error) << "fdom NULL " ;
    }

    NPY<int>* idom = getIDomain();
    if(idom)
    { 
        idom->setQuad(m_settings, 0 );
    } 
    else
    {
        LOG(error) << "idom NULL " ;
    }
    
}


void OpticksDomain::importBuffer()
{
    NPY<float>* fdom = getFDomain();
    assert(fdom);
    m_space_domain = fdom->getQuad_(0);
    m_time_domain = fdom->getQuad_(1);
    m_wavelength_domain = fdom->getQuad_(2);

    if(m_space_domain.w <= 0.)
    {
        LOG(fatal) << "BAD FDOMAIN" ; 
        dump("OpticksDomain::importBuffer");
        assert(0);
    }

    NPY<int>* idom = getIDomain();
    assert(idom);

    m_settings = idom->getQuad_(0); 
    unsigned int maxrec = m_settings.w ; 

    if(maxrec != 10)
        LOG(fatal) 
            << " from idom settings m_maxrec BUT EXPECT 10 " << maxrec 
            ;

    //assert(maxrec == 10);
}



unsigned OpticksDomain::getMaxRng() const {    return m_settings.y ; } 
unsigned OpticksDomain::getMaxBounce() const { return m_settings.z ; } 
unsigned OpticksDomain::getMaxRec() const {    return m_settings.w ; }

void OpticksDomain::setMaxRng(unsigned maxrng) {       m_settings.y = maxrng ; } 
void OpticksDomain::setMaxBounce(unsigned maxbounce) { m_settings.z = maxbounce ; } 
void OpticksDomain::setMaxRec(unsigned maxrec) {       m_settings.w = maxrec ; } 


void OpticksDomain::dump(const char* msg)
{
    LOG(info) << msg 
              << "\n space_domain      " << gformat(m_space_domain)
              << "\n time_domain       " << gformat(m_time_domain)
              << "\n wavelength_domain " << gformat(m_wavelength_domain)
              ;
}


