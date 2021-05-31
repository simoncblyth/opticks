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


#include "BStr.hh"

// npy-

#include "NStep.hpp"
#include "GenstepNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "uif.h"

#include "PLOG.hh"


const plog::Severity GenstepNPY::LEVEL = PLOG::EnvLevel("GenstepNPY", "DEBUG") ; 


GenstepNPY::GenstepNPY(unsigned gentype, unsigned num_step, const char* config, bool is_default ) 
    :  
    m_onestep(new NStep),
    m_gentype(gentype),
    m_num_step(num_step),
    m_config(config ? strdup(config) : NULL),
    m_is_default(is_default),
    m_material(NULL),
    m_npy(NPY<float>::make(num_step, 6, 4)),
    m_step_index(0),
    m_frame(-1,0,0,0),
    m_frame_transform(1.f,0.f,0.f,0.f, 0.f,1.f,0.f,0.f, 0.f,0.f,1.f,0.f, 0.f,0.f,0.f,1.f),
    m_frame_targetted(false),
    m_num_photons_per_g4event(10000)
{
    //assert( m_config ) ;   FabStep have NULL config   
    m_npy->zero();
}


bool GenstepNPY::isDefault() const { return m_is_default ; } 

// used from cfg4-
void GenstepNPY::setNumPhotonsPerG4Event(unsigned int n)
{
    m_num_photons_per_g4event = n ; 
}
unsigned int GenstepNPY::getNumPhotonsPerG4Event() const 
{
    return m_num_photons_per_g4event ;
}
unsigned int GenstepNPY::getNumG4Event() const 
{
    unsigned int num_photons = getNumPhotons();
    unsigned int ppe = m_num_photons_per_g4event ; 
    unsigned int num_g4event ; 
    if(num_photons < ppe)
    {
        num_g4event = 1 ; 
    }
    else
    {
        assert( num_photons % ppe == 0 && "expecting num_photons to be exactly divisible by NumPhotonsPerG4Event " );
        num_g4event = num_photons / ppe ; 
    }
    return num_g4event ; 
}



NStep* GenstepNPY::getOneStep() const 
{
    return m_onestep ; 
}
unsigned GenstepNPY::getNumPhotons() const 
{
    return m_onestep->getNumPhotons(); 
}
void GenstepNPY::setNumPhotons(unsigned num_photons)
{
    m_onestep->setNumPhotons(num_photons); 
}
// used from okg/OpticksGen
void GenstepNPY::setMaterialLine(unsigned ml)
{
    m_onestep->setMaterialLine(ml); 
}
// used from g4ok/G4Opticks::collectDefaultTorchStep
void GenstepNPY::setOriginTrackID(unsigned id)
{
    m_onestep->setOriginTrackID(id); 
}





void GenstepNPY::addActionControl(unsigned long long  action_control)
{
    m_npy->addActionControl(action_control);
}

const char* GenstepNPY::getMaterial() const 
{
    return m_material ; 
}
const char* GenstepNPY::getConfig() const 
{
    return m_config ; 
}

void GenstepNPY::setMaterial(const char* s)
{
    m_material = strdup(s);
}

unsigned GenstepNPY::getNumStep() const 
{
   return m_num_step ;  
}


/*
frame #4: 0x00000001007ce6d9 libNPY.dylib`GenstepNPY::addStep(this=0x0000000108667020, verbose=false) + 57 at GenstepNPY.cpp:56
frame #5: 0x0000000101e2a7d6 libOpticksGeometry.dylib`OpticksGen::makeTorchstep(this=0x0000000108664f50) + 150 at OpticksGen.cc:182
frame #6: 0x0000000101e2a32e libOpticksGeometry.dylib`OpticksGen::initInputGensteps(this=0x0000000108664f50) + 606 at OpticksGen.cc:74
frame #7: 0x0000000101e2a095 libOpticksGeometry.dylib`OpticksGen::init(this=0x0000000108664f50) + 21 at OpticksGen.cc:37
frame #8: 0x0000000101e2a073 libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108664f50, hub=0x0000000105609f20) + 131 at OpticksGen.cc:32
frame #9: 0x0000000101e2a0bd libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108664f50, hub=0x0000000105609f20) + 29 at OpticksGen.cc:33
frame #10: 0x0000000101e27706 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105609f20) + 118 at OpticksHub.cc:96
frame #11: 0x0000000101e27610 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105609f20, ok=0x0000000105421710) + 416 at OpticksHub.cc:81
frame #12: 0x0000000101e277ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105609f20, ok=0x0000000105421710) + 29 at OpticksHub.cc:83
frame #13: 0x0000000103790294 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfedd8, argc=1, argv=0x00007fff5fbfeeb8) + 260 at OKMgr.cc:46
*/

/**
GenstepNPY::addStep
-----------------------

Settings from the quads are passed into the genstep by addStep, which 
is called from OpticksGen::makeTorchstep 

**/

void GenstepNPY::addStep(bool verbose)
{
    bool dummy_frame = isDummyFrame(); // m_frame.x == -1 
    bool target_acquired = dummy_frame ? true : m_frame_targetted ;
    if(!target_acquired) 
    {
         LOG(fatal) << "GenstepNPY::addStep target MUST be set for non-dummy frame " 
                    << " dummy_frame " << dummy_frame
                    << " m_frame_targetted " << m_frame_targetted
                    << brief()
                    ;
    }

    assert(target_acquired);

    assert(m_npy && m_npy->hasData());

    //unsigned int i = m_step_index ; 

    m_onestep->setGenstepType( m_gentype ) ;    

    m_onestep->fillArray(); 

    NPY<float>* one = m_onestep->getArray(); 

    m_npy->add(one); 

    m_step_index++ ; 
}

NPY<float>* GenstepNPY::getNPY() const 
{
    assert( m_step_index == m_num_step && "GenstepNPY is incomplete, must addStep according to declared num_step");
    return m_npy ; 
}



/**
GenstepNPY::setFrameTransform
-------------------------------

Canonically invoked by okg/OpticksGen::targetGenstep 

**/

void GenstepNPY::setFrameTransform(glm::mat4& frame_transform)
{
    m_frame_transform = frame_transform ;
    setFrameTargetted(true);
    updateAfterSetFrameTransform();  // implemented in subclasses such as TorchStepNPY and FabStepNPY 
}

void GenstepNPY::setFrameTransform(const char* s)
{
    std::string ss(s);
    bool flip = true ;  
    glm::mat4 transform = gmat4(ss, flip);
    setFrameTransform(transform);
}

const glm::mat4& GenstepNPY::getFrameTransform() const 
{
    return m_frame_transform ;
}
void GenstepNPY::setFrameTargetted(bool targetted)
{
    m_frame_targetted = targetted ;
}
bool GenstepNPY::isFrameTargetted() const 
{
    return m_frame_targetted ;
} 


void GenstepNPY::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void GenstepNPY::setFrame(unsigned vindex)
{
    m_frame.x = vindex ; 
    m_frame.y = 0 ; 
    m_frame.z = 0 ; 
    m_frame.w = 0 ; 
}
const glm::ivec4& GenstepNPY::getFrame() const 
{
    return m_frame ; 
}
int GenstepNPY::getFrameIndex() const 
{
    return m_frame.x ; 
}

bool GenstepNPY::isDummyFrame() const 
{
    return m_frame.x == -1 ; 
}

std::string GenstepNPY::brief() const 
{
    std::stringstream ss ; 

    ss << "GenstepNPY "
       << " frameIndex " << getFrameIndex()
       << " frameTargetted " << isFrameTargetted()
       << " frameTransform " << gformat(m_frame_transform)
       ;

    return ss.str();
}

