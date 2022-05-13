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

#include <csignal>
#include <iostream>
#include <iomanip>

#include "SStr.hh"
#include "SVec.hh"

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NStep.hpp"
#include "TorchStepNPY.hpp"

#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "OpticksFlags.hh"
#include "OpticksGenstep.hh"
#include "OpticksEvent.hh"
#include "OpticksActionControl.hh"

#include "PLOG.hh"

const plog::Severity OpticksGenstep::LEVEL = PLOG::EnvLevel("OpticksGenstep", "DEBUG"); 


std::string OpticksGenstep::Dump()   // static
{
    int i0 = (int)OpticksGenstep_INVALID ;  
    int i1 = (int)OpticksGenstep_NumType ;  

    std::stringstream ss ; 
    for(int i=i0 ; i < i1 ; i++)
    { 
         unsigned gencode = i ; 
         unsigned flag = OpticksGenstep_::GenstepToPhotonFlag(gencode); 
         ss 
            << " gencode " << std::setw(3) << gencode 
            << " OpticksGenstep_::Name " << std::setw(25) << OpticksGenstep_::Name(gencode)
            << " OpticksGenstep_::GenstepToPhotonFlag " << std::setw(10) << flag 
            << " OpticksPhoton::Flag " << std::setw(20) << OpticksPhoton::Flag(flag)
            << " OpticksPhoton::Abbrev " << std::setw(5) << OpticksPhoton::Abbrev(flag)
            << std::endl
            ;
    } 
    return ss.str(); 
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

/**
OpticksGenstep::getGencode
----------------------------

**/

unsigned OpticksGenstep::getGencode(unsigned idx) const 
{
    int gs00 = m_gs->getInt(idx,0u,0u) ;
    assert( gs00 > 0 );  
    unsigned gencode = gs00 ; 
    return gencode ; 
}

unsigned OpticksGenstep::getNumPhotons(unsigned idx) const 
{
    unsigned gs03 = m_gs->getUInt(idx,0u,3u) ;
    assert( gs03 > 0 );  
    unsigned numPhotons = gs03 ; 
    return numPhotons ; 
}




glm::ivec4 OpticksGenstep::getHdr(unsigned i) const 
{
    glm::ivec4 hdr = m_gs->getQuadI(i,0);
    return hdr ; 
}
glm::vec4 OpticksGenstep::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_gs->getQuad_(i,1);
    return post ; 
}
glm::vec4 OpticksGenstep::getDeltaPositionStepLength(unsigned i) const 
{
    glm::vec4 dpsl = m_gs->getQuad_(i,2);
    return dpsl ; 
}


glm::vec4 OpticksGenstep::getQ3(unsigned i) const  {  return m_gs->getQuad_(i,3); }
glm::vec4 OpticksGenstep::getQ4(unsigned i) const  {  return m_gs->getQuad_(i,4); }
glm::vec4 OpticksGenstep::getQ5(unsigned i) const  {  return m_gs->getQuad_(i,5); }

// union type shifted getters
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

void OpticksGenstep::dump(const char* msg) const
{
    unsigned modulo = 1 ; 
    unsigned margin = 10 ; 
    dump(modulo, margin, msg); 
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



NPY<float>* OpticksGenstep::MakeCandle(unsigned num_photons, unsigned tagoffset ) // static
{
    unsigned gentype = OpticksGenstep_TORCH  ;
    unsigned num_step = 1 ; 
    const char* config = NULL ; 
    TorchStepNPY* ts = new TorchStepNPY(gentype, config);

    glm::mat4 frame_transform(1.0); 
    ts->setFrameTransform(frame_transform);
    for(unsigned i=0 ; i < num_step ; i++) 
    {    
        if(num_photons > 0) ts->setNumPhotons(num_photons);  // otherwise use default
        ts->addStep(); 
    }    
    NPY<float>* gs = ts->getNPY(); 
    bool compute = true ; 
    gs->setBufferSpec(OpticksEvent::GenstepSpec(compute));
    gs->setArrayContentIndex( tagoffset ); 

    OpticksActionControl oac(gs->getActionControlPtr());     
    oac.add(OpticksActionControl::GS_TORCH_);

    return gs ; 
}

/**
OpticksGenstep::MakeInputPhotonCarrier
----------------------------------------

Invoked from G4Opticks::setInputPhotons 

Fabricates OpticksGenstep_EMITSOURCE genstep and arranges for the GPU 
source_buffer to get filled with the input photons similar to::

    okg/OpticksGen::initFromEmitterGensteps (need the setAux ref to input photons)
    okc/OpticksGenstep::MakeCandle 

**/

OpticksGenstep* OpticksGenstep::MakeInputPhotonCarrier(NPY<float>* ip, unsigned tagoffset, int repeat, const char* wavelength, int eventID ) // static
{

    // this needs to follow GtOpticksTool::mutate
    std::vector<int> wnm ; 
    if( wavelength ) SStr::ISplit(wavelength, wnm, ',' );  
    int event_number = eventID ; // is this 0-based ? 
    unsigned override_wavelength_nm = wnm.size() == 0 ? 0 : wnm[event_number % wnm.size()] ; 

    unsigned ip_num = ip->getNumItems(); 
    NPY<float>* ipr = repeat == 0 ? ip : NPY<float>::make_repeat( ip, repeat );     
    unsigned ipr_num = ipr->getNumItems(); 

    LOG(LEVEL) 
        << " wavelength " << wavelength
        << " wnm.size " << wnm.size()
        << " " << SVec<int>::Desc("wnm", wnm )
        << " eventID " << eventID
        << " override_wavelength_nm " << override_wavelength_nm
        ;

    if( override_wavelength_nm > 0 )
    {
        int j = 2 ; 
        int k = 3 ; 
        int l = 0 ; 
        ipr->setAllValue(j, k, l, float(override_wavelength_nm) );
    } 

    LOG(LEVEL) 
        << " tagoffset " << tagoffset
        << " repeat " << repeat 
        << " ip_num " << ip_num
        << " ip " << ip->getShapeString()
        << " ipr_num " << ipr_num
        << " ipr " << ipr->getShapeString()
        ; 
   
    NStep onestep ; 
    onestep.setGenstepType( OpticksGenstep_EMITSOURCE );  
    onestep.setNumPhotons(  ipr_num ); 
    onestep.fillArray(); 
    NPY<float>* gs = onestep.getArray(); 


    bool compute = true ; 
    ipr->setBufferSpec(OpticksEvent::SourceSpec(compute));
    ipr->setArrayContentIndex( tagoffset ); 

    gs->setBufferSpec(OpticksEvent::GenstepSpec(compute));
    gs->setArrayContentIndex( tagoffset ); 

    OpticksActionControl oac(gs->getActionControlPtr());     
    oac.add(OpticksActionControl::GS_EMITSOURCE_);       // needed ?
    LOG(LEVEL) 
        << " gs " << gs 
        << " oac.desc " << oac.desc("gs") 
        << " oac.numSet " << oac.numSet() 
        ; 

    gs->setAux((void*)ipr);  // under-radar association of input photons with the fabricated genstep 

    OpticksGenstep* ogs = new OpticksGenstep(gs);
    return ogs ; 
}
 
