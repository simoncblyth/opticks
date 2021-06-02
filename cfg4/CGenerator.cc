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

#include "Opticks.hh"
#include "OpticksGen.hh"
#include "OpticksGenstep.hh"
#include "OpticksCfg.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

#include "CG4.hh"

#include "NGLM.hpp"
#include "NPY.hpp"
#include "TorchStepNPY.hpp"
#include "NGunConfig.hpp"

#include "CTorchSource.hh"
#include "CGunSource.hh"
#include "CInputPhotonSource.hh"
#include "CPrimarySource.hh"
#include "CGenstepSource.hh"

#include "CDetector.hh"
#include "CGenerator.hh"

#include "PLOG.hh"


const plog::Severity CGenerator::LEVEL = PLOG::EnvLevel("CGenerator", "DEBUG") ; 


CGenerator::CGenerator(OpticksGen* gen, CG4* g4)
    :
    m_gen(gen), 
    m_ok(m_gen->getOpticks()),
    m_cfg(m_ok->getCfg()),
    m_g4(g4),
    m_source_code(m_gen->getSourceCode()),
    m_source_type(OpticksFlags::SourceType(m_source_code)),
    m_gensteps(NULL),
    m_onestep(true),
    m_num_g4evt(1),
    m_photons_per_g4evt(0),
    m_gensteps_per_g4evt(0),
    m_source(initSource(m_source_code))
    // m_source must be last as initSource will change m_gensteps, m_onestep, ...
{
    init();
}

void CGenerator::init()
{
}

CSource* CGenerator::initSource(unsigned code)
{
    LOG(LEVEL) 
        << " code " << code 
        << " SourceType " << OpticksFlags::SourceType(code) 
        << " m_source_type " << m_source_type
        ; 

    CSource* source = NULL ;  

    if(code == OpticksGenstep_G4GUN)
    {
        source = initG4GunSource();
    } 
    else if(code == OpticksGenstep_TORCH)      
    {
        source = initTorchSource();
    } 
    else if(code == OpticksGenstep_EMITSOURCE) 
    {
        source = initInputPhotonSource();
    } 
    else if(code == OpticksGenstep_GENSTEPSOURCE) 
    {
        source = initInputGenstepSource();
    }

    assert(source && "code not expected" ) ;

    LOG(LEVEL) 
        << " code " << code
        << " type " << m_source_type
        << " " << ( m_onestep ? "ONESTEP/DYNAMIC" : "ALLSTEP/STATIC" ) 
        ; 

    return source ; 
}


unsigned CGenerator::getSourceCode() const { return m_source_code ; }
const char* CGenerator::getSourceType() const { return m_source_type ; }
CSource* CGenerator::getSource() const { return m_source ; }
unsigned CGenerator::getNumG4Event() const { return m_num_g4evt ;  }
unsigned CGenerator::getNumPhotonsPerG4Event() const { return m_photons_per_g4evt ;  }
bool CGenerator::isOneStep() const { return m_onestep ; } // formerly m_dynamic
bool CGenerator::hasGensteps() const { return m_gensteps != NULL ; }

NPY<float>* CGenerator::getGensteps() const { return m_gensteps ; }
NPY<float>* CGenerator::getSourcePhotons() const { return m_source->getSourcePhotons() ; }


void CGenerator::setNumG4Event(unsigned num) { m_num_g4evt = num ; }
void CGenerator::setNumPhotonsPerG4Event(unsigned num) { m_photons_per_g4evt = num ; }
void CGenerator::setNumGenstepsPerG4Event(unsigned num) { m_gensteps_per_g4evt = num ; }
void CGenerator::setGensteps(NPY<float>* gensteps) { m_gensteps = gensteps ; }
void CGenerator::setOneStep(bool onestep) { m_onestep = onestep ; }


/**
CGenerator::configureEvent
---------------------------

Invoked from CG4::initEvent/CG4::propagate record 
generator config into the OpticksEvent.

**/

void CGenerator::configureEvent(OpticksEvent* evt)
{
   if(hasGensteps())
   {
        LOG(LEVEL) 
            << " pre-existing gensteps (STATIC RUNNING) "
            << " type " << m_source_type
            ;

        evt->setNumG4Event(getNumG4Event());
        evt->setNumPhotonsPerG4Event(getNumPhotonsPerG4Event()) ; 
        evt->zero();  // static approach requires allocation ahead
    
        //evt->dumpDomains("CGenerator::configureEvent");
    } 
    else
    {
         LOG(LEVEL) 
             << " no genstep (DYNAMIC RUNNING) "
             ;  
    }
}


/**
CGenerator::initTorchSource
----------------------------

Converts TorchStepNPY from hub into CSource for Geant4 consumption

**/

CSource* CGenerator::initTorchSource()
{
    LOG(verbose) << "CGenerator::initTorchSource " ; 

    TorchStepNPY* torch = m_gen->getTorchstep();
    NPY<float>* gs = torch->getNPY() ;  
    if(!gs) LOG(fatal) << " NULL gs " ; 
    assert( gs ); 
    setGensteps( gs );  
    // triggers the event init 

    bool onestep = false ; 
    setOneStep(onestep);   // formerly used setDynamic(false)
    

    setNumG4Event( torch->getNumG4Event()); 
    setNumPhotonsPerG4Event( torch->getNumPhotonsPerG4Event()); 

    int verbosity = m_ok->isDbgTorch() ? m_ok->getVerbosity() : 0 ; 
    CSource* source  = static_cast<CSource*>(new CTorchSource( m_ok, torch, verbosity)); 
    return source ; 
}

/**
CGenerator::initInputPhotonSource
----------------------------------

Hmm : what are the inputGensteps for with inputPhotons ? Placeholder ?

**/

CSource* CGenerator::initInputPhotonSource()
{
    LOG(LEVEL) << "[" ; 
    NPY<float>* inputPhotons = m_gen->getInputPhotons();
    NPY<float>* inputGensteps = m_gen->getInputGensteps();
    GenstepNPY* gsnpy = m_gen->getGenstepNPY();

    assert( inputPhotons );
    assert( inputGensteps );
    assert( gsnpy );

    setGensteps(inputGensteps);

    bool onestep = false ;  
    setOneStep(onestep);   // formerly used setDynamic(false)

    unsigned numPhotonsPerG4Event = gsnpy->getNumPhotonsPerG4Event() ; 
    CInputPhotonSource* cips = new CInputPhotonSource( m_ok, inputPhotons, numPhotonsPerG4Event ) ;

    setNumG4Event( cips->getNumG4Event() );
    setNumPhotonsPerG4Event( cips->getNumPhotonsPerG4Event() );

    CSource* source  = static_cast<CSource*>(cips); 
    LOG(LEVEL) << "]" ; 
    return source ; 
}




/**
CGenerator::initInputGenstepSource
----------------------------------

**/

CSource* CGenerator::initInputGenstepSource()
{
    LOG(info) ; 
    //NPY<float>* dgs = m_gen->getDirectGensteps();
    NPY<float>* dgs = m_gen->getInputGensteps();

    assert( dgs );

    setGensteps(dgs);  //why ?

    bool onestep = false ; 
    setOneStep(onestep);   // formerly used setDynamic(false)

    CGenstepSource* gsrc = new CGenstepSource( m_ok, dgs ) ;

    setNumGenstepsPerG4Event( gsrc->getNumGenstepsPerG4Event() );
    setNumG4Event( gsrc->getNumG4Event() );

    CSource* source  = static_cast<CSource*>(gsrc); 
    return source ; 
}




/**
CGenerator::initG4GunSource
-----------------------------

* setup source based on NGunConfig parse of the G4GunConfig string.
* no gensteps at this stage, they have to be collected from Geant4 : dynamic mode
* geometry info is needed as gunconfig picks target volumes by index

**/

CSource* CGenerator::initG4GunSource()
{
    std::string gunconfig = m_gen->getG4GunConfig() ; // NB via OpticksGun in the hub, not directly from Opticks
    LOG(verbose) << "CGenerator::initG4GunSource " 
               << " gunconfig " << gunconfig
                ; 

    NGunConfig* gc = new NGunConfig();
    gc->parse(gunconfig);

    CDetector* detector = m_g4->getDetector();

    unsigned int frameIndex = gc->getFrame() ;
    unsigned int numTransforms = detector->getNumGlobalTransforms() ;

    if(frameIndex < numTransforms )
    {
        const char* pvname = detector->getPVName(frameIndex);
        LOG(info) << "CGenerator::initG4GunSource "
                       << " frameIndex " << frameIndex 
                       << " numTransforms " << numTransforms 
                       << " pvname " << pvname 
                       ;

        glm::mat4 frame = detector->getGlobalTransform( frameIndex );
        gc->setFrameTransform(frame) ;
    }
    else
    {
        LOG(fatal) << "CGenerator::initG4GunSource gun config frameIndex not in detector"
                   << " frameIndex " << frameIndex
                   << " numTransforms " << numTransforms
                          ;
         assert(0);
    }  

    setGensteps(NULL);  // gensteps must be collected from G4, they cannot be fabricated

    setOneStep(true);   // formerly used setDynamic(true) , so no change 

    setNumG4Event(gc->getNumber()); 
    setNumPhotonsPerG4Event(0); 

    CGunSource* gun = new CGunSource(m_ok ) ;
    gun->configure(gc);      

    CSource* source  = static_cast<CSource*>(gun);
    return source ; 
}

