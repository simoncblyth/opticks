
#include "NPY.hpp"
#include "FabStepNPY.hpp"
#include "TorchStepNPY.hpp"
#include "Parameters.hpp"
#include "GLMFormat.hpp"

#include "GGeo.hh"

#include "OpticksPhoton.h"
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksHub.hh"
#include "OpticksEvent.hh"
#include "OpticksActionControl.hh"

#include "OpticksGen.hh"

#include "PLOG.hh"

OpticksGen::OpticksGen(OpticksHub* hub) 
   :
   m_hub(hub),
   m_ok(hub->getOpticks()),
   m_cfg(m_ok->getCfg()),
   m_lookup(hub->getLookup()),
   m_ggeo(hub->getGGeo()),
   m_torchstep(NULL),
   m_fabstep(NULL),
   m_input_gensteps(NULL)
{
    init() ;
}

void OpticksGen::init()
{
    initInputGensteps();
}


NPY<float>* OpticksGen::getInputGensteps()
{
    return m_input_gensteps ; 
}
void OpticksGen::setInputGensteps(NPY<float>* gs)
{
    m_input_gensteps = gs ;  
    if(gs)  // will be NULL for G4GUN for example
    {
        gs->setBufferSpec(OpticksEvent::GenstepSpec());
    }
}

void OpticksGen::initInputGensteps()
{
    if(m_ok->isNoInputGensteps())
    {
        LOG(warning) << "OpticksGen::initInputGensteps SKIP as isNoInputGensteps " ; 
        return ; 
    } 

    LOG(debug) << "OpticksGen::initInputGensteps" ; 

    unsigned int code = m_ok->getSourceCode();
    NPY<float>* gs = NULL ; 

    if( code == FABRICATED || code == MACHINERY  )
    {
        m_fabstep = makeFabstep();
        gs = m_fabstep->getNPY();
    }
    else if(code == TORCH)
    {
        m_torchstep = makeTorchstep() ;
        gs = m_torchstep->getNPY();
    }
    else if( code == CERENKOV || code == SCINTILLATION || code == NATURAL )
    {
        gs = loadGenstepFile("GS_LOADED,GS_LEGACY");
    }
    else if( code == G4GUN  )
    {
        if(m_ok->existsGenstepPath())
        {
             gs = loadGenstepFile("GS_LOADED");
        }
        else
        {
             std::string path = m_ok->getGenstepPath();
             LOG(warning) <<  "G4GUN running, but no gensteps at " << path 
                          << " LIVE G4 is required to provide the gensteps " 
                          ;
        }
    }
    setInputGensteps(gs);
}


TorchStepNPY* OpticksGen::getTorchstep()   
// needed by CGenerator, full details of the torchstep are used in cfg4-/CTorchSource to 
// duplicate the on GPU generation done by Opticks torchstep.h on the CPU for Geant4  
{
    return m_torchstep ; 
}




void OpticksGen::targetGenstep( GenstepNPY* gs )
{
    // targetted positioning and directioning of the torch requires geometry info, 
    // which is not available within npy- so need to externally setFrameTransform
    // based on integer frame volume index

    if(gs->isFrameTargetted())
    {    
        LOG(info) << "OpticksGen::targetGenstep frame targetted already  " << gformat(gs->getFrameTransform()) ;  
    }    
    else 
    {    
        if(m_ggeo)
        {
            glm::ivec4& iframe = gs->getFrame();
            glm::mat4 transform = m_ggeo->getTransform( iframe.x );
            LOG(debug) << "OpticksGen::targetGenstep setting frame " << iframe.x << " " << gformat(transform) ;  
            gs->setFrameTransform(transform);
        }
        else
        {
            LOG(warning) << "OpticksGen::targetGenstep SKIP AS NO GEOMETRY " ; 
        }
    }    
}


void OpticksGen::setMaterialLine( GenstepNPY* gs )
{
    if(!m_ggeo)
    {
        LOG(warning) << "OpticksGen::setMaterialLine no ggeo, skip setting material line " ;
        return ; 
    }

   // translation from a string name from config into a mat line
   // only depends on the GBndLib being loaded, so no G4 complications
   // just need to avoid trying to translate the matline later

   const char* material = gs->getMaterial() ;
   unsigned int matline = m_ggeo->getMaterialLine(material);
   gs->setMaterialLine(matline);  

   LOG(debug) << "OpticksGen::setMaterialLine"
              << " material " << material 
              << " matline " << matline
              ;
}




FabStepNPY* OpticksGen::makeFabstep()
{
    FabStepNPY* fabstep = new FabStepNPY(FABRICATED, 10, 10 );

    const char* material = m_ok->getDefaultMaterial();
    fabstep->setMaterial(material);

    targetGenstep(fabstep);  // sets frame transform
    setMaterialLine(fabstep);
    fabstep->addActionControl(OpticksActionControl::Parse("GS_FABRICATED"));
    return fabstep ; 
}

TorchStepNPY* OpticksGen::makeTorchstep()
{
    TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep();
    targetGenstep(torchstep);  // sets frame transform
    setMaterialLine(torchstep);
    torchstep->addActionControl(OpticksActionControl::Parse("GS_TORCH"));

    bool torchdbg = m_ok->hasOpt("torchdbg");
    torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

    NPY<float>* gs = torchstep->getNPY();
    if(torchdbg) gs->save("$TMP/torchdbg.npy");

    return torchstep ; 
}

NPY<float>* OpticksGen::loadGenstepFile(const char* label)
{
    NPY<float>* gs = m_ok->loadGenstep();
    if(gs == NULL)
    {
        LOG(fatal) << "OpticksGen::loadGenstepFile FAILED" ;
        m_ok->setExit(true);
        return NULL ; 
    } 
    gs->setLookup(m_lookup);

    int modulo = m_cfg->getModulo();

    Parameters* parameters = gs->getParameters();
    parameters->add<int>("Modulo", modulo );
    if(modulo > 0) 
    {    
        parameters->add<std::string>("genstepOriginal",   gs->getDigestString()  );
        LOG(warning) << "OptickGen::loadGenstepFile applying modulo scaledown " << modulo ;
        gs = NPY<float>::make_modulo(gs, modulo);
        parameters->add<std::string>("genstepModulo",   gs->getDigestString()  );
    }    
    gs->addActionControl(OpticksActionControl::Parse(label));
    return gs ; 
}



