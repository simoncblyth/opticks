#include "Opticks.hh"
#include "OpticksHub.hh"
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

#include "CDetector.hh"
#include "CGenerator.hh"

#include "PLOG.hh"


CGenerator::CGenerator(OpticksHub* hub, CG4* g4)
    :
    m_hub(hub),
    m_ok(m_hub->getOpticks()),
    m_cfg(m_ok->getCfg()),
    m_g4(g4),
    m_source(NULL),
    m_num_g4evt(1),
    m_photons_per_g4evt(0),
    m_gensteps(NULL),
    m_dynamic(true)
{
    init();
}


void CGenerator::init()
{
    unsigned code = m_hub->getSourceCode();
    const char* sourceType = OpticksFlags::SourceType(code);

    LOG(info) << "CGenerator::init" 
              << " code " << code
              << " type " << sourceType
              ; 

    if(     code == G4GUN) setSource(makeG4GunSource());
    else if(code == TORCH) setSource(makeTorchSource());
    else if(code == EMITSOURCE) setSource(makeInputPhotonSource());
    else                   assert(0);
}


void CGenerator::setSource(CSource* source)
{
    m_source = source ; 
}
CSource* CGenerator::getSource()
{
    return m_source ; 
}

void CGenerator::setNumG4Event(unsigned num)
{
    m_num_g4evt = num ;
}
unsigned CGenerator::getNumG4Event()
{
    return m_num_g4evt ;  
}

void CGenerator::setNumPhotonsPerG4Event(unsigned num)
{
    m_photons_per_g4evt = num ;
}
unsigned CGenerator::getNumPhotonsPerG4Event()
{
    return m_photons_per_g4evt ;  
}

void CGenerator::setGensteps(NPY<float>* gensteps)
{
    m_gensteps = gensteps ; 
}
NPY<float>* CGenerator::getGensteps()
{
    return m_gensteps ; 
}
void CGenerator::setDynamic(bool dynamic)
{
   m_dynamic = dynamic ; 
}
bool CGenerator::isDynamic()
{
    return m_dynamic ; 
}


bool CGenerator::hasGensteps()
{
    return m_gensteps != NULL ; 
}

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

CSource* CGenerator::makeTorchSource()
{
    LOG(trace) << "CGenerator::makeTorchSource " ; 

    TorchStepNPY* torch = m_hub->getTorchstep();

    setGensteps( torch->getNPY() );  // sets the number of photons and preps buffers (unallocated)
    setDynamic(false);
    setNumG4Event( torch->getNumG4Event()); 
    setNumPhotonsPerG4Event( torch->getNumPhotonsPerG4Event()); 

    int verbosity = m_ok->isDbgTorch() ? 10 : 0 ; 
    CSource* source  = static_cast<CSource*>(new CTorchSource( m_ok, torch, verbosity)); 
    return source ; 
}


CSource* CGenerator::makeInputPhotonSource()
{
    LOG(info) << "CGenerator::makeInputPhotonSource " ; 
    NPY<float>* inputPhotons = m_hub->getInputPhotons();
    GenstepNPY* gsnpy = m_hub->getGenstepNPY();

    int verbosity = m_ok->isDbgSource() ? 10 : 0 ; 
    CSource* source  = static_cast<CSource*>(new CInputPhotonSource( m_ok, inputPhotons, gsnpy, verbosity)); 
    return source ; 
}


/*
unsigned CGenerator::getNumG4Event()
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

*/





CSource* CGenerator::makeG4GunSource()
{
    std::string gunconfig = m_hub->getG4GunConfig() ; // NB via OpticksGun in the hub, not directly from Opticks
    LOG(trace) << "CGenerator::makeG4GunSource " 
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
        LOG(info) << "CGenerator::makeG4GunSource "
                       << " frameIndex " << frameIndex 
                       << " numTransforms " << numTransforms 
                       << " pvname " << pvname 
                       ;

        glm::mat4 frame = detector->getGlobalTransform( frameIndex );
        gc->setFrameTransform(frame) ;
    }
    else
    {
        LOG(fatal) << "CGenerator::makeG4GunSource gun config frameIndex not in detector"
                   << " frameIndex " << frameIndex
                   << " numTransforms " << numTransforms
                          ;
         assert(0);
    }  

    setGensteps(NULL);  // gensteps must be collected from G4, they cannot be fabricated
    setDynamic(true);
    setNumG4Event(gc->getNumber()); 
    setNumPhotonsPerG4Event(0); 

    int verbosity = m_cfg->hasOpt("g4gundbg") ? 10 : 0 ; 
    CGunSource* gun = new CGunSource(m_ok, verbosity) ;
    gun->configure(gc);      

    CSource* source  = static_cast<CSource*>(gun);
    return source ; 
}




