// cfg4-;cfg4--;ggv-;ggv-pmt-test --cfg4 
// cfg4-;cfg4--;op --cfg4 --g4gun --dbg 

#include "CCfG4.hh"

#include "CTestDetector.hh"
#include "CGDMLDetector.hh"

#include "CPropLib.hh"

#include "Recorder.hh"
#include "Rec.hh"

#include "CPrimaryGeneratorAction.hh"
#include "CSteppingAction.hh"
#include "CTorchSource.hh"
#include "CGunSource.hh"

// npy-
#include "Timer.hpp"
#include "Parameters.hpp"
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


// cg4-
#include "CG4.hh"


#define TIMER(s) \
    { \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }


void CCfG4::init(int argc, char** argv)
{
    m_opticks = new Opticks(argc, argv, "cfg4.log");

    m_opticks->setMode( Opticks::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    m_cache = new GCache(m_opticks);

    m_cfg = m_opticks->getCfg();

    TIMER("init");

    configure(argc, argv);
}


CDetector* CCfG4::configureDetector()
{
    CDetector* detector = NULL ; 
    if(m_cfg->hasOpt("test"))
    {
        LOG(info) << "CCfG4::configureDetector G4 simple test geometry " ; 
        std::string testconfig = m_cfg->getTestConfig();
        GGeoTestConfig* ggtc = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
        detector  = static_cast<CDetector*>(new CTestDetector(m_cache, ggtc)) ; 
    }
    else
    {
        // no options here: will load the .gdml sidecar of the geocache .dae 
        LOG(info) << "CCfG4::configureDetector G4 GDML geometry " ; 
        detector  = static_cast<CDetector*>(new CGDMLDetector(m_cache)) ; 
    }
    return detector ; 
}

CPrimaryGeneratorAction* CCfG4::configureGenerator()
{
    // after CG4::configure as needs G4 optical photons
    CSource* source = NULL ; 

    if(m_opticks->getSourceCode() == TORCH)
    {
        LOG(info) << "CCfG4::configureGenerator TORCH " ; 
        m_torch = m_opticks->makeSimpleTorchStep();
        m_torch->addStep(true); // calls update setting pos,dir,pol using the frame transform and preps the NPY buffer
        m_torch->Summary("CCfG4::configure TorchStepNPY::Summary");

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

        LOG(info) << "CCfG4::configureGenerator G4GUN " ; 
        m_evt->setNumG4Event(1); 
        m_evt->setNumPhotonsPerG4Event(0); 

        int g4gun_verbosity = m_cfg->hasOpt("g4gundbg") ? 10 : 0 ; 
        source  = static_cast<CSource*>(new CGunSource(g4gun_verbosity)); 
    }
    else
    {
         LOG(fatal) << "CCfG4::configureGenerator" 
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

    return new CPrimaryGeneratorAction(source) ;
}


CSteppingAction* CCfG4::configureStepping()
{
    return new CSteppingAction(m_lib, m_recorder, m_rec, m_recorder->getVerbosity()) ;
}


void CCfG4::configure(int argc, char** argv)
{
    m_cfg->commandline(argc, argv);  

    m_geant4 = new CG4 ; 

    m_geant4->configure(argc, argv);
    m_evt = m_opticks->makeEvt();

    m_detector = configureDetector();
    m_lib = m_detector->getPropLib();
    m_geant4->setDetectorConstruction(m_detector);

    CPrimaryGeneratorAction* generator  = configureGenerator();
    m_geant4->setPrimaryGeneratorAction(generator);

    CSteppingAction* stepping = configureStepping(); 
    m_geant4->setSteppingAction(stepping);

    m_geant4->initialize();

    setupDomains();

    TIMER("configure");
}


void CCfG4::setupDomains()
{
    // compression domains set after runManager::Initialize, 
    // as extent only known after detector construction

    m_detector->dumpPV("CCfG4::configure dumpPV");
    m_opticks->setSpaceDomain(m_detector->getCenterExtent());

    m_evt->setTimeDomain(m_opticks->getTimeDomain());  
    m_evt->setWavelengthDomain(m_opticks->getWavelengthDomain()) ; 
    m_evt->setSpaceDomain(m_opticks->getSpaceDomain());

    m_evt->dumpDomains("CCfG4::configure dumpDomains");
}


void CCfG4::interactive(int argc, char** argv)
{
    m_geant4->interactive(argc, argv);
}


void CCfG4::propagate()
{
    unsigned int num_g4event = m_evt->getNumG4Event();
 
    LOG(info) << "CCfG4::propagate"
              << " num_g4event " << m_evt->getNumG4Event()
              << " num_photons " << m_evt->getNumPhotons()
              << " steps_per_photon " << m_evt->getMaxRec()
              << " bounce_max " << m_evt->getBounceMax()
              ; 
    TIMER("_propagate");


    m_geant4->BeamOn(num_g4event);


    TIMER("propagate");
}

void CCfG4::save()
{
    m_evt->save(true);
}

CCfG4::~CCfG4()
{
}









