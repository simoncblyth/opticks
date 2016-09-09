#include <string>

#include "NPY.hpp"
#include "TorchStepNPY.hpp"
#include "Parameters.hpp"

#include "GGeo.hh"

#include "OpticksPhoton.h"
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksHub.hh"
#include "OpticksActionControl.hh"

#include "OpticksGen.hh"

#include "PLOG.hh"

OpticksGen::OpticksGen(OpticksHub* hub) 
   :
   m_hub(hub),
   m_ok(hub->getOpticks()),
   m_cfg(m_ok->getCfg()),
   m_lookup(hub->getLookup()),
   m_ggeo(hub->getGGeo())
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
void OpticksGen::setInputGensteps(NPY<float>* igs)
{
    m_input_gensteps = igs ;  
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

    if(code == TORCH)
    {
        m_torchstep = makeTorchstep() ;
        gs = m_torchstep->getNPY();
        gs->addActionControl(OpticksActionControl::Parse("GS_FABRICATED,GS_TORCH"));
    }
    else if( code == CERENKOV || code == SCINTILLATION || code == NATURAL )
    {
        gs = loadGenstepFile();
        gs->addActionControl(OpticksActionControl::Parse("GS_LOADED,GS_LEGACY"));
    }
    else if( code == G4GUN  )
    {
        if(m_ok->isIntegrated())
        {
             LOG(info) << " integrated G4GUN running, gensteps will be collected from G4 directly " ;  
        }
        else
        {
             LOG(info) << " non-integrated G4GUN running, attempt to load gensteps from file " ;  
             gs = loadGenstepFile();
             gs->addActionControl(OpticksActionControl::Parse("GS_LOADED"));
        }
    }
    setInputGensteps(gs);
}


TorchStepNPY* OpticksGen::getTorchstep()   // needed by CGenerator
{
    return m_torchstep ; 
}


TorchStepNPY* OpticksGen::makeTorchstep()
{
    TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep();

    if(m_ggeo)
    {
        m_ggeo->targetTorchStep(torchstep);   // sets frame transform of the torchstep

        // translation from a string name from config into a mat line
        // only depends on the GBndLib being loaded, so no G4 complications
        // just need to avoid trying to translate the matline later

        const char* material = torchstep->getMaterial() ;
        unsigned int matline = m_ggeo->getMaterialLine(material);
        torchstep->setMaterialLine(matline);  

        LOG(debug) << "OpticksGen::makeGenstepTorch"
                   << " config " << torchstep->getConfig() 
                   << " material " << material 
                   << " matline " << matline
                         ;
    }
    else
    {
        LOG(warning) << "OpticksGen::makeTorchstep no ggeo, skip setting torchstep material line " ;
    } 

    bool torchdbg = m_ok->hasOpt("torchdbg");
    torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 

    if(torchdbg)
    {
        NPY<float>* gs = torchstep->getNPY();
        gs->save("$TMP/torchdbg.npy");
    }

    return torchstep ; 
}


NPY<float>* OpticksGen::loadGenstepFile()
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

    return gs ; 
}




