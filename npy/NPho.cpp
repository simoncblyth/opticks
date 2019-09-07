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

#include <iostream>
#include <iomanip>

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NPho.hpp"

#include "PLOG.hh"

NPho::NPho(NPY<float>* photons) 
    :  
    m_photons(photons),
    m_msk(photons->getMsk()),
    m_num_photons(photons ? photons->getNumItems() : 0 ),
    m_num_msk(m_msk ? m_msk->getNumItems() : 0 )
{
    init();
}

void NPho::init()
{
    assert( m_photons->hasShape(-1,4,4) );

    if(m_msk)
    {
        assert( m_num_msk == m_num_photons ); // the mask is assumed to have been already applied to the photons
    }
}

unsigned NPho::getNumPhotons() const 
{
    return m_num_photons ; 
}


NPY<float>* NPho::getPhotons() const 
{
    return m_photons ; 
}

glm::vec4 NPho::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_photons->getQuad(i,0);
    return post ; 
}

glm::vec4 NPho::getDirectionWeight(unsigned i) const 
{
    glm::vec4 dirw = m_photons->getQuad(i,1);
    return dirw ; 
}

glm::vec4 NPho::getPolarizationWavelength(unsigned i) const 
{
    glm::vec4 polw = m_photons->getQuad(i,2);
    return polw ; 
}

glm::uvec4 NPho::getFlags(unsigned i) const 
{
    glm::uvec4 flgs = m_photons->getQuadU(i,3);
    return flgs ; 
}


std::string NPho::desc() const 
{
    std::stringstream ss ;
    ss << "NPho " << ( m_photons ? m_photons->getShapeString() : "-" ) ; 
    return ss.str();
}



std::string NPho::desc(unsigned i) const 
{
    glm::vec4 post = getPositionTime(i);
    glm::vec4 dirw = getDirectionWeight(i);
    glm::vec4 polw = getPolarizationWavelength(i);
    glm::uvec4 flgs = getFlags(i);

    std::stringstream ss ;
    ss 
        << " i " << std::setw(7) << i 
        << " mski " << std::setw(7) << m_photons->getMskIndex(i) 
        << " post " << std::setw(20) << gpresent(post) 
        << " dirw " << std::setw(20) << gpresent(dirw) 
        << " polw " << std::setw(20) << gpresent(polw) 
        << " flgs " << std::setw(20) << gpresent(flgs) 
        ;

    return ss.str();
}


void NPho::Dump(NPY<float>* ox, unsigned modulo, unsigned margin, const char* msg) 
{
    LOG(info) << msg 
              << " modulo " << modulo
              << " margin " << margin
              << " ox " << ( ox ? "Y" : "NULL" ) 
              ;
 
    if(!ox) return ; 
    NPho ph(ox) ;
    ph.dump(modulo, margin); 
}



void NPho::dump(unsigned modulo, unsigned margin, const char* msg) const
{
    NSlice slice(0, getNumPhotons()) ;

    LOG(info) << msg 
              << " slice " << slice.description()
              << " modulo " << modulo
              << " margin " << margin 
              << " desc " << desc() 
              ; 

    for(unsigned i=slice.low ; i < slice.high ; i += slice.step )
    {
        if(slice.isMargin(i, margin) || i % modulo == 0)
        {
            std::cout << desc(i) << std::endl ; 
        }
    }

}


