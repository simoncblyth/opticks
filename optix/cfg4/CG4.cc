// cfg4-;cfg4--;ggv-;ggv-pmt-test --cfg4 
// cfg4-;cfg4--;op --cfg4 --g4gun --dbg 
// cfg4-;cfg4--;ggv-;ggv-g4gun --dbg

//g4-
#include "G4RunManager.hh"
#include "G4String.hh"

#include "G4VisExecutive.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"

//cg4-
#include "CG4.hh"

#include "ActionInitialization.hh"
#include "OpNovicePhysicsList.hh"

#include "CTestDetector.hh"
#include "CGDMLDetector.hh"
#include "CPropLib.hh"
#include "Recorder.hh"
#include "Rec.hh"

#include "CPrimaryGeneratorAction.hh"
#include "CSteppingAction.hh"
#include "CTorchSource.hh"
#include "CGunSource.hh"

// ggeo-
#include "GCache.hh"
#include "GGeoTestConfig.hh"

// optickscore-
#include "Opticks.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"

// npy-
#include "Timer.hpp"
#include "NumpyEvt.hpp"
#include "TorchStepNPY.hpp"
#include "NGunConfig.hpp"
#include "NLog.hpp"

//opticks-
#include "Opticks.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"

//ggeo-
#include "GCache.hh"
#include "GBndLib.hh"
#include "GGeoTestConfig.hh"



#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }

void CG4::init()
{
    m_cfg = m_opticks->getCfg();
    m_cache = new GCache(m_opticks);
    m_evt = m_opticks->makeEvt();

    TIMER("init");
}

void CG4::configure(int argc, char** argv)
{
    m_cfg->commandline(argc, argv);
    m_g4ui = m_cfg->hasOpt("g4ui") ; 

    LOG(info) << "CG4::configure"
              << " g4ui " << m_g4ui
              ; 

    m_runManager = new G4RunManager;

    configurePhysics();
    configureDetector();
    configureGenerator();
    configureStepping();

    TIMER("configure");
}

void CG4::initialize()
{
    LOG(info) << "CG4::initialize" ;

    m_runManager->SetUserInitialization(new ActionInitialization(m_pga, m_sa)) ;
    m_runManager->Initialize();

    setupCompressionDomains();

    m_uiManager = G4UImanager::GetUIpointer();
    m_uiManager->ApplyCommand("/OpNovice/phys/verbose 0");

    LOG(info) << "CG4::initialize DONE" ;

    TIMER("initialize");
}

void CG4::interactive(int argc, char** argv)
{
    if(!m_g4ui) return ; 

    LOG(info) << "CG4::interactive proceeding " ; 

    m_visManager = new G4VisExecutive;
    m_visManager->Initialize();

    m_ui = new G4UIExecutive(argc, argv);
    m_ui->SessionStart();
}

void CG4::propagate()
{
    unsigned int num_g4event = m_evt->getNumG4Event();
 
    LOG(info) << "CG4::propagate"
              << " num_g4event " << m_evt->getNumG4Event()
              << " num_photons " << m_evt->getNumPhotons()
              << " steps_per_photon " << m_evt->getMaxRec()
              << " bounce_max " << m_evt->getBounceMax()
              ; 
    TIMER("_propagate");

    m_runManager->BeamOn(num_g4event);

    TIMER("propagate");
}

void CG4::save()
{
    m_evt->save(true);
}


void CG4::configurePhysics()
{
    OpNovicePhysicsList* npl = new OpNovicePhysicsList();
    //npl->SetVerbose(0);  nope processes not instanciated at this stage

    m_runManager->SetUserInitialization(npl);
    TIMER("configurePhysics");
}

void CG4::configureDetector()
{
    CDetector* detector = NULL ; 
    if(m_cfg->hasOpt("test"))
    {
        LOG(info) << "CG4::configureDetector G4 simple test geometry " ; 
        std::string testconfig = m_cfg->getTestConfig();
        GGeoTestConfig* ggtc = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
        detector  = static_cast<CDetector*>(new CTestDetector(m_cache, ggtc)) ; 
    }
    else
    {
        // no options here: will load the .gdml sidecar of the geocache .dae 
        LOG(info) << "CG4::configureDetector G4 GDML geometry " ; 
        detector  = static_cast<CDetector*>(new CGDMLDetector(m_cache)) ; 
    }

    m_detector = detector ; 
    m_lib = detector->getPropLib();
    m_runManager->SetUserInitialization(detector);

    TIMER("configureDetector");
}


void CG4::configureGenerator()
{
    CSource* source = NULL ; 

    if(m_opticks->getSourceCode() == TORCH)
    {
        LOG(info) << "CG4::configureGenerator TORCH " ; 
        m_torch = m_opticks->makeSimpleTorchStep();
        m_torch->addStep(true); // calls update setting pos,dir,pol using the frame transform and preps the NPY buffer
        m_torch->Summary("CG4::configure TorchStepNPY::Summary");

        m_evt->setGenstepData( m_torch->getNPY() );  // sets the number of photons and preps buffers (unallocated)
        m_evt->setNumG4Event(m_torch->getNumG4Event()); 
        m_evt->setNumPhotonsPerG4Event(m_torch->getNumPhotonsPerG4Event()); 


        int torch_verbosity = m_cfg->hasOpt("torchdbg") ? 10 : 0 ; 
        source  = static_cast<CSource*>(new CTorchSource(m_torch, torch_verbosity)); 
    }
    else if(m_opticks->getSourceCode() == G4GUN)
    {
        // hmm this is G4 only, so should it be arranged at this level  ?
        // without the setGenstepData the evt is not allocated 

        LOG(info) << "CG4::configureGenerator G4GUN " ; 

        std::string gunconfig = m_cfg->getG4GunConfig();
        NGunConfig* gc = new NGunConfig( gunconfig.empty() ? NULL : gunconfig.c_str() );

        m_evt->setNumG4Event(100); 
        m_evt->setNumPhotonsPerG4Event(0); 

        int g4gun_verbosity = m_cfg->hasOpt("g4gundbg") ? 10 : 0 ; 
        source  = static_cast<CSource*>(new CGunSource(gc, g4gun_verbosity)); 
    }
    else
    {
         LOG(fatal) << "CG4::configureGenerator" 
                    << " expecting TORCH or G4GUN " 
                    ; 
         assert(0);
    }


    int stepping_verbosity = m_cfg->hasOpt("steppingdbg") ? 10 : 0 ; 
    // recorder is back here in order to pass to source for primary recording (unused?)
    m_recorder = new Recorder(m_lib, m_evt, stepping_verbosity ); 
    m_rec = new Rec(m_lib, m_evt) ; 
    if(m_cfg->hasOpt("primary"))
         m_recorder->setupPrimaryRecording();

    source->setRecorder(m_recorder);

    m_pga = new CPrimaryGeneratorAction(source) ;
    TIMER("configureGenerator");
}



void CG4::configureStepping()
{
    m_sa = new CSteppingAction(m_lib, m_recorder, m_rec, m_recorder->getVerbosity()) ;
    TIMER("configureStepping");
}

void CG4::setupCompressionDomains()
{
    m_detector->dumpPV("CG4::setupCompressionDomains dumpPV");
    m_opticks->setSpaceDomain(m_detector->getCenterExtent());

    m_evt->setTimeDomain(m_opticks->getTimeDomain());  
    m_evt->setWavelengthDomain(m_opticks->getWavelengthDomain()) ; 
    m_evt->setSpaceDomain(m_opticks->getSpaceDomain());

    m_evt->dumpDomains("CG4::setupCompressionDomains");
}



CG4::~CG4()
{
    delete m_runManager;
}



