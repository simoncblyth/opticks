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


#include "NMeta.hpp"

#include "NCSG.hpp"
#include "NPY.hpp"
#include "FabStepNPY.hpp"
#include "TorchStepNPY.hpp"
#include "NEmitPhotonsNPY.hpp"

#include "GLMFormat.hpp"

#include "GGeoBase.hh"
#include "GGeo.hh"
#include "GBndLib.hh"

#include "OpticksVersionNumber.hh"
#include "OpticksGun.hh"
#include "OpticksPhoton.h"
#include "OpticksGenstep.hh"
#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksCfg.hh"
#include "OpticksHub.hh"
#include "OpticksEvent.hh"
#include "OpticksActionControl.hh"

#include "OpticksGen.hh"

#include "PLOG.hh"

const plog::Severity OpticksGen::LEVEL = PLOG::EnvLevel("OpticksGen", "DEBUG") ; 


NPY<float>* OpticksGen::getInputGensteps() const { return m_direct_gensteps ? m_direct_gensteps : m_legacy_gensteps ; }
NPY<float>* OpticksGen::getInputPhotons() const {   return m_input_photons ; }




OpticksGen::OpticksGen(OpticksHub* hub) 
    :
    m_hub(hub),
    m_gun(new OpticksGun(hub)),
    m_ok(hub->getOpticks()),
    m_cfg(m_ok->getCfg()),
    m_ggeo(hub->getGGeo()),
    m_ggb(hub->getGGeoBase()),
    m_blib(m_ggb->getBndLib()),
    m_lookup(hub->getLookup()),
    m_torchstep(NULL),
    m_fabstep(NULL),
    m_csg_emit(hub->findEmitter()),
    m_dbgemit(m_ok->isDbgEmit()),     // --dbgemit
    m_emitter(m_csg_emit ? new NEmitPhotonsNPY(m_csg_emit, OpticksGenstep_EMITSOURCE, m_ok->getSeed(), m_dbgemit, m_ok->getMaskBuffer(), m_ok->getGenerateOverride() ) : NULL ),
    m_input_photons(NULL),
    m_tagoffset(0),
    m_direct_gensteps(m_ok->findGensteps(m_tagoffset)),
    m_legacy_gensteps(NULL),
    m_source_code(initSourceCode())
{
    init() ;
}

Opticks* OpticksGen::getOpticks() const { return m_ok ; }
std::string OpticksGen::getG4GunConfig() const { return m_gun->getConfig() ; }


/**
This is trying to treat direct and legacy gensteps the same ... but they aint

**/


unsigned OpticksGen::initSourceCode() const 
{
    unsigned code = 0 ; 
    if(m_direct_gensteps)
    {
        code = OpticksGenstep_GENSTEPSOURCE ; 
    }  
    else if(m_emitter) 
    {
        code = OpticksGenstep_EMITSOURCE ; 
    }
    else
    { 
        code = m_ok->getSourceCode()  ;
    }
    return code ; 
}

unsigned OpticksGen::getSourceCode() const 
{
    return m_source_code ;  
}

/**
OpticksGen::init
------------------

Upshot is that one of the below gets set

* m_direct_gensteps 
* m_legacy_gensteps : for emitter as well as legacy gensteps

**/

void OpticksGen::init()
{
    LOG(LEVEL); 
    if(m_direct_gensteps)
    {
        initFromDirectGensteps();
    }  
    else if(m_emitter) 
    {
        initFromEmitterGensteps();
    }
    else
    { 
        initFromLegacyGensteps();
    }
}

void OpticksGen::initFromEmitterGensteps()
{
    // emitter bits and pieces get dressed up 
    // perhaps make a class to do this ?   

    LOG(LEVEL); 
    NPY<float>* iox = m_emitter->getPhotons();  // these photons maybe masked 
    setInputPhotons(iox);

    m_fabstep = m_emitter->getFabStep();

    NPY<float>* gs = m_emitter->getFabStepData();
    assert( gs );

    gs->setAux((void*)iox); // under-radar association of input photons with the fabricated genstep

    // this gets picked up by OpticksRun::setGensteps 

    const char* oac_ = "GS_EMITSOURCE" ;  

    gs->addActionControl(OpticksActionControl::Parse(oac_));

    OpticksActionControl oac(gs->getActionControlPtr());
    setLegacyGensteps(gs);

    LOG(LEVEL) 
        << "getting input photons and shim genstep "
        << " --dbgemit " << m_dbgemit
        << " input_photons " << m_input_photons->getNumItems()
        << " oac : " << oac.description("oac") 
        ; 
}



void OpticksGen::initFromDirectGensteps()
{
    assert( m_direct_gensteps ) ; 
    std::string loadpath = m_direct_gensteps->getMeta<std::string>("loadpath",""); 
    LOG(LEVEL) << loadpath ; 
    m_direct_gensteps->setBufferSpec(OpticksEvent::GenstepSpec(m_ok->isCompute()));
}

void OpticksGen::initFromLegacyGensteps()
{
    LOG(LEVEL); 
    if(m_ok->isNoInputGensteps() || m_ok->isEmbedded())
    {
        LOG(warning) << "SKIP as isNoInputGensteps OR isEmbedded  " ; 
        return ; 
    } 

    const char* type = m_ok->getSourceType();
    unsigned code = m_ok->getSourceCode();

    LOG(LEVEL) 
        << " code " << code
        << " type " << type
        ;

    NPY<float>* gs = makeLegacyGensteps(code) ; 
    assert( gs );
    setLegacyGensteps(gs);
}


/**
OpticksGen::makeLegacyGensteps
-------------------------------

Legacy gensteps can be FABRICATED, MACHINERY or TORCH 
and are created directly OR they can be CERENKOV, SCINTILLATION, NATURAL, 
G4GUN which are loaded fro files.

**/

NPY<float>* OpticksGen::makeLegacyGensteps(unsigned code)
{
    NPY<float>* gs = NULL ; 

    const char* srctype = OpticksFlags::SourceType( code ) ; 
    assert( srctype ); 

    LOG(LEVEL) 
       << " code " << code
       << " srctype " << srctype  
       ;
 

    if( code == OpticksGenstep_FABRICATED || code == OpticksGenstep_MACHINERY  )
    {
        m_fabstep = makeFabstep();
        gs = m_fabstep->getNPY();
    }
    else if(code == OpticksGenstep_TORCH)
    {
        m_torchstep = makeTorchstep(code) ;
        gs = m_torchstep->getNPY();
    }
    else if( code == OpticksGenstep_G4Cerenkov_1042 || code == OpticksGenstep_DsG4Scintillation_r3971 || code == OpticksGenstep_NATURAL )
    {
        gs = loadLegacyGenstepFile("GS_LOADED,GS_LEGACY");
    }
    else if( code == OpticksGenstep_G4GUN  )
    {
        if(m_ok->existsLegacyGenstepPath())
        {
             gs = loadLegacyGenstepFile("GS_LOADED");
        }
        else
        {
             std::string path = m_ok->getLegacyGenstepPath();
             LOG(warning) <<  "G4GUN running, but no gensteps at " << path 
                          << " LIVE G4 is required to provide the gensteps " 
                          ;
        }
    }
    return gs ; 
}




FabStepNPY* OpticksGen::getFabStep() const { return m_fabstep ; }


GenstepNPY* OpticksGen::getGenstepNPY() const 
{
    unsigned source_code = getSourceCode();

    GenstepNPY* gsnpy = NULL ; 

    if(source_code == OpticksGenstep_TORCH)
    {
        gsnpy = dynamic_cast<GenstepNPY*>(getTorchstep()) ;
    } 
    else if( source_code == OpticksGenstep_EMITSOURCE )
    {
        gsnpy = dynamic_cast<GenstepNPY*>(getFabStep());
    }
    return gsnpy ; 
}


void OpticksGen::setLegacyGensteps(NPY<float>* gs)
{
    m_legacy_gensteps = gs ;  
    if(gs)  // will be NULL for G4GUN for example
    {
        gs->setBufferSpec(OpticksEvent::GenstepSpec(m_ok->isCompute()));
    }
}

void OpticksGen::setInputPhotons(NPY<float>* ox)
{
    m_input_photons = ox ;  
    if(ox) 
    {
        LOG(LEVEL) 
            << " ox " << ox->getShapeString()
            << " ox.hasMsk " << ( ox->hasMsk() ? "Y" : "N" )
            ;

        ox->setBufferSpec(OpticksEvent::SourceSpec(m_ok->isCompute()));
    }
}


TorchStepNPY* OpticksGen::getTorchstep() const // used by CGenerator for  cfg4-/CTorchSource duplication
{
    return m_torchstep ; 
}



/**
OpticksGen::targetGenstep
---------------------------

Targetted positioning and directioning of the torch requires geometry info, 
which is not available within npy- so need to externally setFrameTransform
based on integer frame volume index.

Observe that with "--machinery" option the node_index from 
GenstepNPY::getFrame (iframe.x) is staying at -1. So have 
added a reset to 0 for this. See:

* notes/issues/opticks-t-2020-oct7-2of434-fails-eventTest-OpSeederTest.rst

**/

void OpticksGen::targetGenstep( GenstepNPY* gs )
{
    if(gs->isFrameTargetted())
    {    
        LOG(LEVEL) << "frame targetted already  " << gformat(gs->getFrameTransform()) ;  
    }    
    else 
    {   
        if(m_hub)
        {
            const glm::ivec4& iframe = gs->getFrame();

            int node_index = iframe.x ; 
            if(node_index < 0)
            {
                LOG(fatal) << "node_index from GenstepNPY is -1 (dummy frame), resetting to 0" ; 
                node_index = 0 ; 
            }

            glm::mat4 transform = m_ggeo->getTransform( node_index );
            LOG(LEVEL) << "setting frame " << iframe.x << " " << gformat(transform) ;  
            gs->setFrameTransform(transform);
        }
        else
        {
            LOG(error) << "SKIP AS NO GEOMETRY " ; 
        }

    }    
}




/**
OpticksGen::setMaterialLine
-----------------------------

Translation from a string name from config into a mat line
only depends on the GBndLib being loaded, so no G4 complications
just need to avoid trying to translate the matline later.

**/

void OpticksGen::setMaterialLine( GenstepNPY* gs )
{
    if(!m_blib)
    {
        LOG(warning) << "no blib, skip setting material line " ;
        return ; 
    }
   const char* material = gs->getMaterial() ;

   if(material == NULL)
      LOG(fatal) << "NULL material from GenstepNPY, probably missed material in torch config" ;
   assert(material);

   unsigned int matline = m_blib->getMaterialLine(material);
   gs->setMaterialLine(matline);  

   LOG(debug) << "OpticksGen::setMaterialLine"
              << " material " << material 
              << " matline " << matline
              ;
}


FabStepNPY* OpticksGen::makeFabstep()
{
    FabStepNPY* fabstep = new FabStepNPY(OpticksGenstep_FABRICATED, 10, 10 );

    const char* material = m_ok->getDefaultMaterial();
    fabstep->setMaterial(material);

    targetGenstep(fabstep);  // sets frame transform
    setMaterialLine(fabstep);
    fabstep->addActionControl(OpticksActionControl::Parse("GS_FABRICATED"));
    return fabstep ; 
}



TorchStepNPY* OpticksGen::makeTorchstep(unsigned gencode)
{
    assert( gencode == OpticksGenstep_TORCH ); 

    TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep(gencode);

    if(torchstep->isDefault())
    {
        int frameIdx = torchstep->getFrameIndex(); 
        int detectorDefaultFrame = m_ok->getDefaultFrame() ; 
        int gdmlaux_target =  m_ggeo ? m_ggeo->getFirstNodeIndexForGDMLAuxTargetLVName() : -1 ;  // sensitive to GDML auxilary lvname metadata (label, target) 
        int cmdline_target = m_ok->getGenstepTarget() ;   // --gensteptarget

        unsigned active_target = 0 ;
           
        if( cmdline_target > 0 )
        {
            active_target = cmdline_target ;  
        }
        else if( gdmlaux_target > 0 )
        {
            active_target = gdmlaux_target ;  
        }
        
        LOG(error) 
            << " as torchstep isDefault replacing placeholder frame " 
            << " frameIdx : " << frameIdx
            << " detectorDefaultFrame : " << detectorDefaultFrame
            << " cmdline_target [--gensteptarget] : " << cmdline_target
            << " gdmlaux_target : " << gdmlaux_target
            << " active_target : " << active_target
            ; 

        torchstep->setFrame(active_target); 
    }


    targetGenstep(torchstep);  // sets frame transform
    setMaterialLine(torchstep);
    torchstep->addActionControl(OpticksActionControl::Parse("GS_TORCH"));

    unsigned num_photons0 = torchstep->getNumPhotons(); 
    int generateoverride = m_ok->getGenerateOverride(); 

    if( generateoverride > 0)
    { 
        LOG(error) << " overriding number of photons with --generateoverride " ; 
        torchstep->setNumPhotons( generateoverride ); 
    }
    unsigned num_photons = torchstep->getNumPhotons(); 

    LOG(error)
        << " generateoverride " << generateoverride
        << " num_photons0 " << num_photons0
        << " num_photons " << num_photons
        ; 


    bool torchdbg = m_ok->hasOpt("torchdbg");
    torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

    NPY<float>* gs = torchstep->getNPY();
    gs->setArrayContentVersion(-OPTICKS_VERSION_NUMBER) ; 

    if(torchdbg) gs->save("$TMP/torchdbg.npy");

    return torchstep ; 
}

NPY<float>* OpticksGen::loadLegacyGenstepFile(const char* label)
{
    NPY<float>* gs = m_ok->loadLegacyGenstep();
    if(gs == NULL)
    {
        LOG(fatal) << "OpticksGen::loadLegacyGenstepFile FAILED" ;
        m_ok->setExit(true);
        return NULL ; 
    } 
    gs->setLookup(m_lookup);

    int modulo = m_cfg->getModulo();

    NMeta* parameters = gs->getParameters();
    parameters->add<int>("Modulo", modulo );
    if(modulo > 0) 
    {    
        parameters->add<std::string>("genstepOriginal",   gs->getDigestString()  );
        LOG(warning) << "OptickGen::loadLegacyGenstepFile applying modulo scaledown " << modulo ;
        gs = NPY<float>::make_modulo(gs, modulo);
        parameters->add<std::string>("genstepModulo",   gs->getDigestString()  );
    }    
    gs->addActionControl(OpticksActionControl::Parse(label));
    return gs ; 
}


