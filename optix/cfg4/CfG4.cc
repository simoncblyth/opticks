// cfg4-
#include "CfG4.hh"
#include "PhysicsList.hh"
#include "Detector.hh"
#include "ActionInitialization.hh"
#include "Recorder.hh"

// npy-
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
    m_cfg = new OpticksCfg<Opticks>("opticks", m_opticks,false);
    m_cache = new GCache(m_prefix, "cfg4.log", "info");
}

void CfG4::configure(int argc, char** argv)
{
    m_cache->configure(argc, argv);  // logging setup needs to happen before below general config
    m_cfg->commandline(argc, argv);  

    bool test = m_cfg->hasOpt("test") ;
    unsigned int code = m_opticks->getSourceCode(); // cfg is lodged inside opticks
    assert( test && code == TORCH && "cfg4 only supports source type TORCH with test geometries" );

    std::string typ = Opticks::SourceTypeLowercase(code);
    std::string tag = m_cfg->getEventTag();
    std::string cat = m_cfg->getEventCat();

    unsigned int steps_per_photon = m_cfg->getRecordMax();
    unsigned int photons_per_g4event = m_cfg->getNumPhotonsPerG4Event() ;


    m_torch = m_opticks->makeSimpleTorchStep();

    m_torch->setNumPhotonsPerG4Event(photons_per_g4event);
    m_num_photons = m_torch->getNumPhotons();
    m_num_g4event = m_torch->getNumG4Event();

    if(strcmp(tag.c_str(), "-5") == 0)  m_torch->setIncidentSphereSPolarized(true) ;


    // TODO: move event metadata handling/persisting into NumpyEvt
    m_recorder = new Recorder(typ.c_str(),tag.c_str(),cat.c_str(),m_num_photons,steps_per_photon, photons_per_g4event); 

    LOG(info) << "CfG4::configure" 
              << " typ " << typ
              << " tag " << tag 
              << " cat " << cat
              << " num_g4event " << m_num_g4event 
              << " num_photons " << m_num_photons
              << " steps_per_photon " << steps_per_photon
              ;


    std::string testconfig = m_cfg->getTestConfig();
    m_testconfig = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );


    m_detector  = new Detector(m_cache, m_testconfig) ; 

    m_runManager = new G4RunManager;

    m_runManager->SetUserInitialization(new PhysicsList());
    m_runManager->SetUserInitialization(m_detector);
    ActionInitialization* ai = new ActionInitialization(m_recorder, m_torch) ;
    m_runManager->SetUserInitialization(ai);
    m_runManager->Initialize();

    // domains used for record compression 
    m_recorder->setCenterExtent(m_detector->getCenterExtent());
    m_recorder->setBoundaryDomain(m_detector->getBoundaryDomain());
}
void CfG4::propagate()
{
    LOG(info) << "CfG4::propagate"
              << " num_g4event " << m_num_g4event 
              << " num_photons " << m_num_photons 
              ; 
    m_runManager->BeamOn(m_num_g4event);
}

void CfG4::save()
{
    m_recorder->save();
}
CfG4::~CfG4()
{
    delete m_runManager;
}



