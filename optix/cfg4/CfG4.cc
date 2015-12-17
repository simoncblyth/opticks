// cfg4-
#include "CfG4.hh"
#include "PhysicsList.hh"
#include "DetectorConstruction.hh"
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

//g4-
#include "G4RunManager.hh"
//#include "G4UImanager.hh"
#include "G4String.hh"
//#include "G4UIExecutive.hh"

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

    bool constituents ; 
    m_blib = GBndLib::load(m_cache, constituents=true);
    m_blib->Summary("CfG4::configure");

    unsigned int code = m_opticks->getSourceCode(); // cfg is lodged inside opticks
    assert(code == TORCH && "cfg4 only supports source type TORCH" );


    m_torch = m_opticks->makeSimpleTorchStep();
    m_torch->dump();

    m_num_photons = m_torch->getNumPhotons();
 
    std::string typ = Opticks::SourceTypeLowercase(code);
    std::string tag = m_cfg->getEventTag();
    std::string cat = m_cfg->getEventCat();

    unsigned int maxrec = m_cfg->getRecordMax();
    assert(maxrec == 10);

    m_g4_photons_per_event = m_cfg->getG4PhotonsPerEvent();
    assert( m_num_photons % m_g4_photons_per_event == 0 && "expecting num_photons to be exactly divisible by g4_photons_per_event" );

    m_g4_nevt = m_num_photons / m_g4_photons_per_event ; 

    LOG(info) << "CfG4::configure" 
              << " typ " << typ
              << " tag " << tag 
              << " cat " << cat
              << " m_g4_nevt " << m_g4_nevt 
              << " m_g4_photons_per_event " << m_g4_photons_per_event
              << " m_num_photons " << m_num_photons
              << " maxrec " << maxrec
              ;

    m_recorder = new Recorder(typ.c_str(),tag.c_str(),cat.c_str(),m_num_photons,maxrec, m_g4_photons_per_event); 
    if(strcmp(tag.c_str(), "-5") == 0)  m_recorder->setIncidentSphereSPolarized(true) ;

    m_detector  = new DetectorConstruction() ; 
    m_runManager = new G4RunManager;

    m_runManager->SetUserInitialization(new PhysicsList());
    m_runManager->SetUserInitialization(m_detector);
    m_runManager->SetUserInitialization(new ActionInitialization(m_recorder));
    m_runManager->Initialize();

    // domains used for record compression 
    m_recorder->setCenterExtent(m_detector->getCenterExtent());
    m_recorder->setBoundaryDomain(m_detector->getBoundaryDomain());
}
void CfG4::propagate()
{
    LOG(info) << "CfG4::propagate"
              << " g4_nevt " << m_g4_nevt 
              << " m_g4_photons_per_event " << m_g4_photons_per_event
              << " num_photons " << m_num_photons 
              ; 
    m_runManager->BeamOn(m_g4_nevt);
}

void CfG4::save()
{
    m_recorder->save();
}
CfG4::~CfG4()
{
    delete m_runManager;
}



