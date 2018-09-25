#include "Opticks.hh"
#include "OpticksGen.hh"
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



CGenerator::CGenerator(OpticksGen* gen, CG4* g4)
    :
    m_gen(gen), 
    m_ok(m_gen->getOpticks()),
    m_cfg(m_ok->getCfg()),
    m_g4(g4),
    m_source_code(m_gen->getSourceCode()),
    m_gensteps(NULL),
    m_dynamic(true),
    m_num_g4evt(1),
    m_photons_per_g4evt(0),
    m_gensteps_per_g4evt(0),
    m_source(initSource(m_source_code))
    // m_source must be last as initSource will change m_gensteps, m_dynamic, ...
{
    init();
}

void CGenerator::init()
{
}

CSource* CGenerator::initSource(unsigned code)
{
    const char* sourceType = OpticksFlags::SourceType(code);

    CSource* source = NULL ;  

    if(     code == G4GUN)      source = initG4GunSource();
    else if(code == TORCH)      source = initTorchSource();
    else if(code == EMITSOURCE) source = initInputPhotonSource();
    else if(code == GENSTEPSOURCE) source = initInputGenstepSource();
    else  assert( 0 && "code not handled" ); 
 
    assert(source) ;

    LOG(fatal) 
        << " code " << code
        << " type " << sourceType
        << " " << ( m_dynamic ? "DYNAMIC" : "STATIC" ) 
        ; 

    return source ; 
}


unsigned CGenerator::getSourceCode() const { return m_source_code ; }
CSource* CGenerator::getSource() const { return m_source ; }
unsigned CGenerator::getNumG4Event() const { return m_num_g4evt ;  }
unsigned CGenerator::getNumPhotonsPerG4Event() const { return m_photons_per_g4evt ;  }
bool CGenerator::isDynamic() const { return m_dynamic ; }
bool CGenerator::hasGensteps() const { return m_gensteps != NULL ; }

NPY<float>* CGenerator::getGensteps() const { return m_gensteps ; }
NPY<float>* CGenerator::getSourcePhotons() const { return m_source->getSourcePhotons() ; }


void CGenerator::setNumG4Event(unsigned num) { m_num_g4evt = num ; }
void CGenerator::setNumPhotonsPerG4Event(unsigned num) { m_photons_per_g4evt = num ; }
void CGenerator::setNumGenstepsPerG4Event(unsigned num) { m_gensteps_per_g4evt = num ; }
void CGenerator::setGensteps(NPY<float>* gensteps) { m_gensteps = gensteps ; }
void CGenerator::setDynamic(bool dynamic) { m_dynamic = dynamic ; }


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
        LOG(info) << "CGenerator:configureEvent"
                  << " fabricated TORCH genstep (STATIC RUNNING) "
                  ;

        evt->setNumG4Event(getNumG4Event());
        evt->setNumPhotonsPerG4Event(getNumPhotonsPerG4Event()) ; 
        evt->zero();  // static approach requires allocation ahead
    
        //evt->dumpDomains("CGenerator::configureEvent");
    } 
    else
    {
         LOG(info) << "CGenerator::configureEvent"
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

    setDynamic(false);
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
    LOG(info) << "CGenerator::initInputPhotonSource " ; 
    NPY<float>* inputPhotons = m_gen->getInputPhotons();
    NPY<float>* inputGensteps = m_gen->getInputGensteps();
    GenstepNPY* gsnpy = m_gen->getGenstepNPY();

    assert( inputPhotons );
    assert( inputGensteps );
    assert( gsnpy );

    setGensteps(inputGensteps);
    setDynamic(false);

    CInputPhotonSource* cips = new CInputPhotonSource( m_ok, inputPhotons, gsnpy ) ;

    setNumG4Event( cips->getNumG4Event() );
    setNumPhotonsPerG4Event( cips->getNumPhotonsPerG4Event() );

    CSource* source  = static_cast<CSource*>(cips); 
    return source ; 
}




/**
CGenerator::initInputGenstepSource
----------------------------------

**/

CSource* CGenerator::initInputGenstepSource()
{
    LOG(info) << "CGenerator::initInputGenstepSource " ; 
    //NPY<float>* dgs = m_gen->getDirectGensteps();
    NPY<float>* dgs = m_gen->getInputGensteps();

    assert( dgs );

    setGensteps(dgs);  //why ?
    setDynamic(false);

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
    setDynamic(true);
    setNumG4Event(gc->getNumber()); 
    setNumPhotonsPerG4Event(0); 

    CGunSource* gun = new CGunSource(m_ok ) ;
    gun->configure(gc);      

    CSource* source  = static_cast<CSource*>(gun);
    return source ; 
}

