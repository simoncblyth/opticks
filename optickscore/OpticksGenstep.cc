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

#include "OpticksPhoton.h"
#include "OpticksFlags.hh"
#include "OpticksGenstep.hh"

#include "PLOG.hh"



const char* OpticksGenstep::Invalid_                 = "Invalid" ;
const char* OpticksGenstep::G4Cerenkov_1042_         = "G4Cerenkov_1042" ;
const char* OpticksGenstep::G4Scintillation_1042_    = "G4Scintillation_1042" ;
const char* OpticksGenstep::DsG4Cerenkov_r3971_      = "DsG4Cerenkov_r3971" ;
const char* OpticksGenstep::DsG4Scintillation_r3971_ = "DsG4Scintillation_r3971" ;

const char* OpticksGenstep::Gentype(int gentype)
{
    const char* s = 0 ;
    switch(gentype)
    {
        case OpticksGenstep_Invalid:                 s=Invalid_                  ; break ; 
        case OpticksGenstep_G4Cerenkov_1042:         s=G4Cerenkov_1042_          ; break ; 
        case OpticksGenstep_G4Scintillation_1042:    s=G4Scintillation_1042_     ; break ; 
        case OpticksGenstep_DsG4Cerenkov_r3971:      s=DsG4Cerenkov_r3971_       ; break ; 
        case OpticksGenstep_DsG4Scintillation_r3971: s=DsG4Scintillation_r3971_  ; break ; 
        case OpticksGenstep_NumType:                 s=Invalid_                  ; break ; 
        default:                                     s=Invalid_  ;
    }
    return s;
}


OpticksGenstep::OpticksGenstep(const NPY<float>* gs) 
    :  
    m_gs(gs)
{
    init();
}

void OpticksGenstep::init()
{
    assert( m_gs->hasShape(-1,6,4) );
}

const NPY<float>* OpticksGenstep::getGensteps() const { return m_gs ; }

int OpticksGenstep::getContentVersion() const { return m_gs ? m_gs->getArrayContentVersion() : 0 ; }
unsigned OpticksGenstep::getNumGensteps() const { return m_gs ? m_gs->getNumItems() : 0 ; }
unsigned OpticksGenstep::getNumPhotons() const { return m_gs ? m_gs->getUSum(0,3) : 0  ; }

float OpticksGenstep::getAvgPhotonsPerGenstep() const { 

   float num_photons = getNumPhotons();
   float num_gensteps = getNumGensteps() ; 
   return num_gensteps > 0 ? num_photons/num_gensteps : 0 ; 
}

std::string OpticksGenstep::desc() const 
{
    std::stringstream ss ;
    ss << "OpticksGenstep " 
       << ( m_gs ? m_gs->getShapeString() : "-" ) 
       << " content_version " << getContentVersion()
       << " num_gensteps " << getNumGensteps()
       << " num_photons " << getNumPhotons()
       << " avg_photons_per_genstep " << getAvgPhotonsPerGenstep()
       ; 
    return ss.str();
}


unsigned OpticksGenstep::getGencode(unsigned idx) const 
{
    int gs00 = m_gs->getInt(idx,0u,0u) ;

    int gencode = -1 ; 

    int content_version = getContentVersion() ; 

    if( content_version == 0 )  // old style unversioned gensteps , this is fallback when no metadata 
    {
        gencode = gs00 < 0 ? CERENKOV : SCINTILLATION ;  
    }
    else if( content_version >= 1042 )   // G4 version starting point 
    {
        gencode = gs00 ; 
    }
    else if( content_version <= -10 )   // OK version starting point 
    {
        gencode = gs00 ; 
    }
    else
    { 
        LOG(fatal) << " unexpected gensteps content_version " << content_version ; 
        assert(0); 
    }

    bool expected = gencode == CERENKOV || gencode == SCINTILLATION  ; 

    if(!expected)
         LOG(fatal) << "unexpected gencode " 
                    << " gencode " << gencode
                    << " content_version " << content_version
                    << " flag " << OpticksFlags::Flag(gencode) 
                    ;

    assert(expected) ; 
    return gencode ; 
}




glm::ivec4 OpticksGenstep::getHdr(unsigned i) const 
{
    glm::ivec4 hdr = m_gs->getQuadI(i,0);
    return hdr ; 
}
glm::vec4 OpticksGenstep::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_gs->getQuad(i,1);
    return post ; 
}
glm::vec4 OpticksGenstep::getDeltaPositionStepLength(unsigned i) const 
{
    glm::vec4 dpsl = m_gs->getQuad(i,2);
    return dpsl ; 
}


glm::vec4 OpticksGenstep::getQ3(unsigned i) const  {  return m_gs->getQuad(i,3); }
glm::vec4 OpticksGenstep::getQ4(unsigned i) const  {  return m_gs->getQuad(i,4); }
glm::vec4 OpticksGenstep::getQ5(unsigned i) const  {  return m_gs->getQuad(i,5); }

glm::ivec4 OpticksGenstep::getI3(unsigned i) const  {  return m_gs->getQuadI(i,3); }
glm::ivec4 OpticksGenstep::getI4(unsigned i) const  {  return m_gs->getQuadI(i,4); }
glm::ivec4 OpticksGenstep::getI5(unsigned i) const  {  return m_gs->getQuadI(i,5); }


std::string OpticksGenstep::desc(unsigned i) const 
{
    glm::ivec4 hdr = getHdr(i);
    glm::vec4 post = getPositionTime(i);
    glm::vec4 dpsl = getDeltaPositionStepLength(i);

    std::stringstream ss ;
    ss 
        << " i " << std::setw(7) << i 
        << " hdr " << std::setw(20) << gpresent(hdr) 
        << " post " << std::setw(20) << gpresent(post) 
        << " dpsl " << std::setw(20) << gpresent(dpsl) 
        ;

    return ss.str();
}


void OpticksGenstep::Dump(const NPY<float>* gs_, unsigned modulo, unsigned margin, const char* msg) 
{
    LOG(info) << msg 
              << " modulo " << modulo
              << " margin " << margin
              << " gs_ " << ( gs_ ? "Y" : "NULL" ) 
              ;
 
    if(!gs_) return ; 
    OpticksGenstep gs(gs_) ;
    gs.dump(modulo, margin); 
}


void OpticksGenstep::dump(unsigned modulo, unsigned margin, const char* msg) const
{
    NSlice slice(0, getNumGensteps()) ;

    LOG(info) << msg 
              << " slice " << slice.description()
              << " modulo " << modulo
              << " margin " << margin 
              << std::endl 
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


