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

#include "BStr.hh"
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NPho.hpp"

#include "PLOG.hh"

/**

The mask gives access to origin photon indices.

**/

NPho* NPho::Make( NPY<float>* photons, const char* opt) // static 
{
    NPho* pho = new NPho(opt); 
    pho->setPhotons(photons); 
    return pho ; 
}


NPho::NPho(const char* opt) 
    :  
    m_photons(NULL),
    m_msk(NULL),
    m_num_photons(0),
    m_num_msk(0), 
    m_num_quad(0),
    m_mski(opt ? BStr::Contains(opt, "mski", ',' ) : true ),
    m_post(opt ? BStr::Contains(opt, "post", ',' ) : true ),
    m_dirw(opt ? BStr::Contains(opt, "dirw", ',' ) : true ),
    m_polw(opt ? BStr::Contains(opt, "polw", ',' ) : true ),
    m_flgs(opt ? BStr::Contains(opt, "flgs", ',' ) : true )
{
    init();
}


void NPho::setPhotons(NPY<float>* photons)
{
    m_photons = photons ;
    assert( m_photons );  
    assert( m_photons->hasShape(-1,4,4) );
    m_num_photons = m_photons->getNumItems() ;    
    m_num_quad = photons->getShape(1) ; 
    assert( m_photons->hasShape(-1,m_num_quad,4) );

    m_msk = m_photons->getMsk();
    if(m_msk)
    {
        m_num_msk = m_msk->getNumItems() ; 
        assert( m_num_msk == m_num_photons ); // the mask is assumed to have been already applied to the photons
    }
}

void NPho::init()
{
}

unsigned NPho::getNumPhotons() const 
{
    return m_num_photons ; 
}

/**
NPho::getNumQuad
-------------------

* Usually 4 from with photons array shape: (num_photons, *4*, 4)   
* Sometimes 2 with hiy array shape: (num_hiy, 2, 4) 

**/

unsigned NPho::getNumQuad() const 
{
    return m_num_quad ; 
}


NPY<float>* NPho::getPhotons() const 
{
    return m_photons ; 
}

glm::vec4 NPho::getQ0(unsigned i) const 
{
    glm::vec4 q0 = m_photons->getQuad_(i,0);
    return q0 ; 
}
glm::vec4 NPho::getQ1(unsigned i) const 
{
    glm::vec4 q1 = m_photons->getQuad_(i,1);
    return q1 ; 
}

glm::vec4 NPho::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_photons->getQuad_(i,0);
    return post ; 
}

glm::vec4 NPho::getDirectionWeight(unsigned i) const 
{
    glm::vec4 dirw = m_photons->getQuad_(i,1);
    return dirw ; 
}

glm::vec4 NPho::getPolarizationWavelength(unsigned i) const 
{
    assert( m_num_quad >= 1 );
    glm::vec4 polw = m_photons->getQuad_(i,2); 
    return polw ; 
}

glm::ivec4 NPho::getFlags(unsigned i) const 
{
    unsigned qlast = m_num_quad - 1 ; 
    glm::ivec4 flgs = m_photons->getQuadI(i,qlast);  // union type shifting getter
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
    std::stringstream ss ;
    ss << " i " << std::setw(7) << i ; 

    if(m_num_quad == 4)
    {
        glm::vec4 post = getPositionTime(i);
        glm::vec4 dirw = getDirectionWeight(i);
        glm::vec4 polw = getPolarizationWavelength(i);
        glm::ivec4 flgs = getFlags(i);

        if(m_mski) ss << " mski " << std::setw(7) << m_photons->getMskIndex(i)  ; 
        if(m_post) ss << " post " << std::setw(20) << gpresent(post) ;
        if(m_dirw) ss << " dirw " << std::setw(20) << gpresent(dirw) ;
        if(m_polw) ss << " polw " << std::setw(20) << gpresent(polw) ;
        if(m_flgs) ss << " flgs " << std::setw(20) << gpresent(flgs) ;
    }
    else if(m_num_quad == 2)
    {
        glm::vec4 q0 = getQ0(i);
        glm::vec4 q1 = getQ1(i);
        glm::ivec4 flgs = getFlags(i);
 
        ss << " q0 " << std::setw(20) << gpresent(q0)  ; 
        ss << " q1 " << std::setw(20) << gpresent(q1)  ; 
        ss << " flgs " << std::setw(20) << gpresent(flgs)  ; 
    }
    return ss.str();
}





void NPho::Dump(NPY<float>* ox, unsigned modulo, unsigned margin, const char* opt) 
{
    LOG(info) << opt
              << " modulo " << modulo
              << " margin " << margin
              << " ox " << ( ox ? "Y" : "NULL" ) 
              ;
 
    if(!ox) return ; 
    NPho ph(opt) ;
    ph.setPhotons(ox); 
    ph.dump(modulo, margin); 
}

void NPho::Dump(NPY<float>* ox, unsigned maxDump, const char* opt) 
{
    LOG(info) << opt
              << " ox " << ( ox ? "Y" : "NULL" ) 
              ;
 
    if(!ox) return ; 
    NPho ph(opt) ;
    ph.setPhotons(ox); 
    ph.dump("NPho::Dump", maxDump); 
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



void NPho::dump(const char* msg, unsigned maxDump) const 
{
    unsigned numPhotons = getNumPhotons() ;
    unsigned numDump = maxDump > 0 ?  std::min( numPhotons, maxDump ) : numPhotons ; 
    LOG(info) << msg 
              << " desc " << desc() 
              << " numPhotons " << numPhotons
              << " maxDump " << maxDump
              << " numDump " << numDump
              ; 

    for(unsigned i=0 ; i < numDump  ; i++)
    {
        std::cout << desc(i) << std::endl ;
    }
}


