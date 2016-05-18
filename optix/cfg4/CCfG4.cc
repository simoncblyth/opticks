// cfg4-
#include "CCfG4.hh"

#include "CDetector.hh"
#include "CPropLib.hh"

#include "Recorder.hh"
#include "Rec.hh"
#include "PrimaryGeneratorAction.hh"
#include "SteppingAction.hh"
#include "CSource.hh"

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
    
    m_opticks->setMode( Opticks::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    m_cache = new GCache(m_opticks);

    m_cfg = m_opticks->getCfg();

    TIMER("init");

}

void CCfG4::configure(int argc, char** argv)
{
    m_cfg->commandline(argc, argv);  

    assert( m_cfg->hasOpt("test") && m_opticks->getSourceCode() == TORCH && "cfg4 only supports source type TORCH with test geometries" );

    std::string testconfig = m_cfg->getTestConfig();
    m_testconfig = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
    m_detector  = new CDetector(m_cache, m_testconfig) ; 
    CPropLib* clib = m_detector->getPropLib() ;

    m_evt = m_opticks->makeEvt();

    Parameters* params = m_evt->getParameters() ;
    params->add<std::string>("cmdline", m_cfg->getCommandLine() );

    m_torch = m_opticks->makeSimpleTorchStep();
    m_torch->addStep(true); // calls update setting pos,dir,pol using the frame transform and preps the NPY buffer
    m_torch->Summary("CCfG4::configure TorchStepNPY::Summary");

    m_evt->setGenstepData( m_torch->getNPY() );  // sets the number of photons and preps buffers (unallocated)
    m_num_g4event = m_torch->getNumG4Event();
    m_num_photons = m_evt->getNumPhotons();

    unsigned int photons_per_g4event = m_torch->getNumPhotonsPerG4Event();
    
    int stepping_verbosity = m_cfg->hasOpt("steppingdbg") ? 10 : 0 ; 
    int generator_verbosity = m_cfg->hasOpt("torchdbg") ? 10 : 0 ; 

    m_recorder = new Recorder(m_evt , photons_per_g4event, stepping_verbosity > 0 ); 
    if(m_cfg->hasOpt("primary"))
        m_recorder->setupPrimaryRecording();
   
    CSource* generator = new CSource(m_torch, m_recorder);
    generator->SetVerbosity(generator_verbosity);

    m_rec = new Rec(clib, m_evt) ; 

    m_geant4 = new CG4 ; 
    m_geant4->configure(argc, argv);
    m_geant4->setDetectorConstruction(m_detector);
    m_geant4->setPrimaryGeneratorAction(new PrimaryGeneratorAction(generator)) ;
    m_geant4->setSteppingAction(new SteppingAction(clib, m_recorder, m_rec, stepping_verbosity));

    m_geant4->initialize();


    m_recorder->setPropLib(clib);

    // compression domains set after runManager::Initialize, 
    // as extent only known after detector construction

    m_opticks->setSpaceDomain(m_detector->getCenterExtent());

    m_evt->setTimeDomain(m_opticks->getTimeDomain());  
    m_evt->setWavelengthDomain(m_opticks->getWavelengthDomain()) ; 
    m_evt->setSpaceDomain(m_opticks->getSpaceDomain());

    m_evt->dumpDomains("CCfG4::configure dumpDomains");
    m_detector->dumpPV("CCfG4::configure dumpPV");

    TIMER("configure");
}


void CCfG4::interactive(int argc, char** argv)
{
    m_geant4->interactive(argc, argv);
}


void CCfG4::propagate()
{
    LOG(info) << "CCfG4::propagate"
              << " num_g4event " << m_num_g4event 
              << " num_photons " << m_num_photons 
              << " steps_per_photon " << m_evt->getMaxRec()
              << " bounce_max " << m_evt->getBounceMax()
              ; 
    TIMER("_propagate");

    m_geant4->BeamOn(m_num_g4event);


    TIMER("propagate");
}

void CCfG4::save()
{
    m_evt->save(true);
}

CCfG4::~CCfG4()
{
}









