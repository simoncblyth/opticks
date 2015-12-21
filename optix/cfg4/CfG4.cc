// cfg4-
#include "CfG4.hh"
#include "PhysicsList.hh"
#include "Detector.hh"
#include "ActionInitialization.hh"
#include "Recorder.hh"

// npy-
#include "NumpyEvt.hpp"
#include "TorchStepNPY.hpp"
#include "NLog.hpp"

//opticks-
#include "Opticks.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"

//ggeo-
#include "GCache.hh"
#include "GBndLib.hh"
#include "GGeoTestConfig.hh"

//g4-
#include "G4RunManager.hh"
#include "G4String.hh"

void CfG4::init()
{
    m_opticks = new Opticks();
    m_cfg = m_opticks->getCfg();
    m_cache = new GCache(m_prefix, "cfg4.log", "info");
}

void CfG4::configure(int argc, char** argv)
{
    m_cache->configure(argc, argv);  // logging setup needs to happen before below general config
    m_cfg->commandline(argc, argv);  
    assert( m_cfg->hasOpt("test") && m_opticks->getSourceCode() == TORCH && "cfg4 only supports source type TORCH with test geometries" );

    std::string testconfig = m_cfg->getTestConfig();
    m_testconfig = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
    m_detector  = new Detector(m_cache, m_testconfig) ; 

    m_evt = m_opticks->makeEvt();

    m_torch = m_opticks->makeSimpleTorchStep();
    m_torch->addStep(true); // calls update setting pos,dir,pol using the frame transform and preps the NPY buffer

    m_evt->setGenstepData( m_torch->getNPY() );  // sets the number of photons and preps buffers (unallocated)
    m_num_g4event = m_torch->getNumG4Event();
    m_num_photons = m_evt->getNumPhotons();

    unsigned int photons_per_g4event = m_torch->getNumPhotonsPerG4Event();

    m_recorder = new Recorder(m_evt , photons_per_g4event); 

    m_runManager = new G4RunManager;
    m_runManager->SetUserInitialization(new PhysicsList());
    m_runManager->SetUserInitialization(m_detector);
    m_runManager->SetUserInitialization(new ActionInitialization(m_recorder, m_torch)) ;
    m_runManager->Initialize();

    // compression domains set after runManager::Initialize, 
    // as extent only known after detector construction

    m_opticks->setSpaceDomain(m_detector->getCenterExtent());

    m_evt->setTimeDomain(m_opticks->getTimeDomain());  
    m_evt->setWavelengthDomain(m_opticks->getWavelengthDomain()) ; 
    m_evt->setSpaceDomain(m_opticks->getSpaceDomain());

    m_evt->dumpDomains("CfG4::configure dumpDomains");
}

void CfG4::propagate()
{
    LOG(info) << "CfG4::propagate"
              << " num_g4event " << m_num_g4event 
              << " num_photons " << m_num_photons 
              << " steps_per_photon " << m_evt->getMaxRec()
              ; 
    m_runManager->BeamOn(m_num_g4event);
}

void CfG4::save()
{
    m_evt->save(true);
}

CfG4::~CfG4()
{
    delete m_runManager;
}

