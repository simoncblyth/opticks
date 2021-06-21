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

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NStep.hpp"
#include "TorchStepNPY.hpp"

#include "OpticksPhoton.h"
#include "OpticksFlags.hh"
#include "OpticksGenstep.hh"
#include "OpticksEvent.hh"
#include "OpticksActionControl.hh"

#include "PLOG.hh"

const plog::Severity OpticksGenstep::LEVEL = PLOG::EnvLevel("OpticksGenstep", "DEBUG"); 

const char* OpticksGenstep::INVALID_                 = "INVALID" ;
const char* OpticksGenstep::G4Cerenkov_1042_         = "G4Cerenkov_1042" ;
const char* OpticksGenstep::G4Scintillation_1042_    = "G4Scintillation_1042" ;
const char* OpticksGenstep::DsG4Cerenkov_r3971_      = "DsG4Cerenkov_r3971" ;
const char* OpticksGenstep::DsG4Scintillation_r3971_ = "DsG4Scintillation_r3971" ;
const char* OpticksGenstep::DsG4Scintillation_r4695_ = "DsG4Scintillation_r4695" ;
const char* OpticksGenstep::TORCH_                   = "torch" ;
const char* OpticksGenstep::FABRICATED_              = "fabricated" ;
const char* OpticksGenstep::EMITSOURCE_              = "emitsource" ;
const char* OpticksGenstep::NATURAL_                 = "natural" ;
const char* OpticksGenstep::MACHINERY_               = "machinery" ;
const char* OpticksGenstep::G4GUN_                   = "g4gun" ;
const char* OpticksGenstep::PRIMARYSOURCE_           = "primarysource" ;
const char* OpticksGenstep::GENSTEPSOURCE_           = "genstepsource" ;



std::string OpticksGenstep::Dump()   // static
{
    int i0 = (int)OpticksGenstep_INVALID ;  
    int i1 = (int)OpticksGenstep_NumType ;  

    std::stringstream ss ; 
    for(int i=i0 ; i < i1 ; i++)
    { 
         unsigned gencode = i ; 
         unsigned flag = OpticksGenstep::GenstepToPhotonFlag(gencode); 
         ss 
            << " gencode " << std::setw(3) << gencode 
            << " OpticksGenstep::Gentype " << std::setw(25) << Gentype(gencode)
            << " OpticksGenstep::GenstepToPhotonFlag " << std::setw(10) << flag 
            << " OpticksFlags::Flag " << std::setw(20) << OpticksFlags::Flag(flag)
            << " OpticksFlags::Abbrev " << std::setw(5) << OpticksFlags::Abbrev(flag)
            << std::endl
            ;
    } 
    return ss.str(); 
}



const char* OpticksGenstep::Gentype(int gentype)  // static
{
    const char* s = 0 ;
    switch(gentype)
    {
        case OpticksGenstep_INVALID:                 s=INVALID_                  ; break ; 
        case OpticksGenstep_G4Cerenkov_1042:         s=G4Cerenkov_1042_          ; break ; 
        case OpticksGenstep_G4Scintillation_1042:    s=G4Scintillation_1042_     ; break ; 
        case OpticksGenstep_DsG4Cerenkov_r3971:      s=DsG4Cerenkov_r3971_       ; break ; 
        case OpticksGenstep_DsG4Scintillation_r3971: s=DsG4Scintillation_r3971_  ; break ; 
        case OpticksGenstep_DsG4Scintillation_r4695: s=DsG4Scintillation_r4695_  ; break ; 
        case OpticksGenstep_TORCH:                   s=TORCH_                    ; break ; 
        case OpticksGenstep_FABRICATED:              s=FABRICATED_               ; break ; 
        case OpticksGenstep_EMITSOURCE:              s=EMITSOURCE_               ; break ; 
        case OpticksGenstep_NATURAL:                 s=NATURAL_                  ; break ; 
        case OpticksGenstep_MACHINERY:               s=MACHINERY_                ; break ; 
        case OpticksGenstep_G4GUN:                   s=G4GUN_                    ; break ; 
        case OpticksGenstep_PRIMARYSOURCE:           s=PRIMARYSOURCE_            ; break ; 
        case OpticksGenstep_GENSTEPSOURCE:           s=GENSTEPSOURCE_            ; break ; 
        case OpticksGenstep_NumType:                 s=INVALID_                  ; break ; 
        default:                                     s=INVALID_  ;
    }
    return s;
}

unsigned OpticksGenstep::SourceCode(const char* type) // static
{
    unsigned int code = OpticksGenstep_INVALID  ; 
    if(     strcmp(type,G4Cerenkov_1042_)==0)          code = OpticksGenstep_G4Cerenkov_1042 ;
    else if(strcmp(type,G4Scintillation_1042_)==0)     code = OpticksGenstep_G4Scintillation_1042 ;
    else if(strcmp(type,DsG4Cerenkov_r3971_ )==0)      code = OpticksGenstep_DsG4Cerenkov_r3971 ;
    else if(strcmp(type,DsG4Scintillation_r3971_)==0)  code = OpticksGenstep_DsG4Scintillation_r3971 ;
    else if(strcmp(type,DsG4Scintillation_r4695_)==0)  code = OpticksGenstep_DsG4Scintillation_r4695 ;
    else if(strcmp(type,TORCH_  )==0)                  code = OpticksGenstep_TORCH ;
    else if(strcmp(type,FABRICATED_)==0)               code = OpticksGenstep_FABRICATED ;
    else if(strcmp(type,EMITSOURCE_)==0)               code = OpticksGenstep_EMITSOURCE ;
    else if(strcmp(type,NATURAL_)==0)                  code = OpticksGenstep_NATURAL ;
    else if(strcmp(type,MACHINERY_)==0)                code = OpticksGenstep_MACHINERY ;
    else if(strcmp(type,G4GUN_)==0)                    code = OpticksGenstep_G4GUN ;
    else if(strcmp(type,PRIMARYSOURCE_)==0)            code = OpticksGenstep_PRIMARYSOURCE ;
    else if(strcmp(type,GENSTEPSOURCE_)==0)            code = OpticksGenstep_GENSTEPSOURCE ;
    return code ; 
}

bool OpticksGenstep::IsValid(int gentype)   // static 
{
   const char* s = Gentype(gentype); 
   bool invalid = strcmp(s, INVALID_) == 0 ;   
   return !invalid ; 
}

bool OpticksGenstep::IsCerenkov(int gentype)  // static
{
   return gentype == OpticksGenstep_G4Cerenkov_1042  || gentype == OpticksGenstep_DsG4Cerenkov_r3971 ; 
}
bool OpticksGenstep::IsScintillation(int gentype)  // static
{
   return gentype == OpticksGenstep_G4Scintillation_1042 || gentype == OpticksGenstep_DsG4Scintillation_r3971 || gentype == OpticksGenstep_DsG4Scintillation_r4695 ; 
}
bool OpticksGenstep::IsTorchLike(int gentype)   // static
{
   return gentype == OpticksGenstep_TORCH || gentype == OpticksGenstep_FABRICATED || gentype == OpticksGenstep_EMITSOURCE ; 
}  
bool OpticksGenstep::IsEmitSource(int gentype)   // static
{
   return gentype == OpticksGenstep_EMITSOURCE ; 
}  
bool OpticksGenstep::IsMachinery(int gentype)  // static
{
   return gentype == OpticksGenstep_MACHINERY ; 
}


/**
OpticksGenstep::GenstepToPhotonFlag
-------------------------------------

Translate gentype from Genstep to Photon.

Used by CG4Ctx::setEvent 

**/

unsigned OpticksGenstep::GenstepToPhotonFlag(int gentype)  // static
{
    unsigned phcode = 0 ;  
    if(!OpticksGenstep::IsValid(gentype))
    {
        LOG(fatal) << "invalid gentype " << gentype ; 
        phcode = NAN_ABORT ; 
    }
    else if(OpticksGenstep::IsCerenkov(gentype))
    {
        phcode = CERENKOV ; 
    }
    else if(OpticksGenstep::IsScintillation(gentype))
    {
        phcode = SCINTILLATION ; 
    }
    else if(OpticksGenstep::IsTorchLike(gentype))
    {
        phcode = TORCH ; 
    }
    else
    {
        LOG(fatal) << "unexpected gentype " << gentype ; 
        phcode = NAN_ABORT ; 
    }
    return phcode ;   
}

unsigned OpticksGenstep::GentypeToPhotonFlag(char gentype)  // static
{
    unsigned phcode = 0 ;  
    switch(gentype)
    {
        case 'C': phcode = CERENKOV          ; break ; 
        case 'S': phcode = SCINTILLATION     ; break ; 
        case 'T': phcode = TORCH             ; break ;  
        default:  phcode = NAN_ABORT         ; break ; 
    }

    if( phcode == NAN_ABORT) 
    {
        LOG(fatal) 
           << "unexpected gentype " << gentype 
           << " phcode " << phcode
           ;
    }
    return phcode ;   
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

OpticksGenstep* OpticksGenstep::MakeInputPhotonCarrier(NPY<float>* ip, unsigned tagoffset, int repeat ) // static
{
    unsigned ip_num = ip->getNumItems(); 
    NPY<float>* ipr = repeat == 0 ? ip : NPY<float>::make_repeat( ip, repeat );     
    unsigned ipr_num = ipr->getNumItems(); 

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
 
