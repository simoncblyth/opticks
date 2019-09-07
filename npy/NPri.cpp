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
#include "NPri.hpp"

#include "PLOG.hh"


unsigned NPri::getNumG4Event() const { return 1 ; }

NPri::NPri(const NPY<float>* primaries) 
    :  
    m_primaries(primaries)
{
    init();
}

void NPri::init()
{
    assert( m_primaries->hasShape(-1,4,4) );
}

unsigned NPri::getNumPri() const 
{
    return m_primaries->getNumItems() ; 
}

const NPY<float>* NPri::getPrimaries() const 
{
    return m_primaries ; 
}

glm::vec4 NPri::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_primaries->getQuad(i,0);
    return post ; 
}

glm::vec4 NPri::getDirectionWeight(unsigned i) const 
{
    glm::vec4 dirw = m_primaries->getQuad(i,1);
    return dirw ; 
}

glm::vec4 NPri::getPolarizationKineticEnergy(unsigned i) const 
{
    glm::vec4 polk = m_primaries->getQuad(i,2);
    return polk ; 
}

glm::ivec4 NPri::getFlags(unsigned i) const   // doesnt do the union trick so -ves are mangled
{
    glm::ivec4 flgs = m_primaries->getQuadI(i,3);
    return flgs ; 
}

int NPri::getEventIndex(unsigned i) const {   return m_primaries->getInt(i, 3, 0, 0) ; }
int NPri::getVertexIndex(unsigned i) const {  return m_primaries->getInt(i, 3, 0, 1) ; }
int NPri::getParticleIndex(unsigned i) const { return m_primaries->getInt(i, 3, 0, 2) ; } 
int NPri::getPDGCode(unsigned i) const {      return m_primaries->getInt(i, 3, 0, 3) ; } 


std::string NPri::desc() const 
{
    std::stringstream ss ;
    ss << "NPri " << ( m_primaries ? m_primaries->getShapeString() : "-" ) ; 
    return ss.str();
}



std::string NPri::desc(unsigned i) const 
{
    glm::vec4 post = getPositionTime(i);
    glm::vec4 dirw = getDirectionWeight(i);
    glm::vec4 polw = getPolarizationKineticEnergy(i);
    glm::ivec4 flgs = getFlags(i);

    std::stringstream ss ;
    ss 
        << " i " << std::setw(7) << i 
        << " post " << std::setw(20) << gpresent(post) 
        << " dirw " << std::setw(20) << gpresent(dirw) 
        << " polw " << std::setw(20) << gpresent(polw) 
        << " flgs " << std::setw(20) << gpresent(flgs) 
        ;

    return ss.str();
}


void NPri::Dump(NPY<float>* primaries, unsigned modulo, unsigned margin, const char* msg) 
{
    LOG(info) << msg 
              << " modulo " << modulo
              << " margin " << margin
              << " primaries " << ( primaries ? "Y" : "NULL" ) 
              ;
 
    if(!primaries) return ; 
    NPri pri(primaries) ;
    pri.dump(modulo, margin); 
}



void NPri::dump(unsigned modulo, unsigned margin, const char* msg) const
{
    NSlice slice(0, getNumPri()) ;

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


