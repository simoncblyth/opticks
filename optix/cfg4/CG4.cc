// cfg4-;cfg4--;ggv-;ggv-pmt-test --cfg4 
// cfg4-;cfg4--;op --cfg4 --g4gun --dbg 
// cfg4-;cfg4--;ggv-;ggv-g4gun --dbg

//g4-
#include "G4RunManager.hh"
#include "G4String.hh"

#include "G4VisExecutive.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4GeometryManager.hh"

//cg4-
#include "CG4.hh"

#include "ActionInitialization.hh"

#ifdef OLDPHYS
#include "PhysicsList.hh"
#else
#include "OpNovicePhysicsList.hh"
#endif

#include "CTestDetector.hh"
#include "CGDMLDetector.hh"
#include "CPropLib.hh"
#include "CRecorder.hh"
#include "Rec.hh"
#include "CStepRec.hh"

#include "CPrimaryGeneratorAction.hh"
#include "CSteppingAction.hh"
#include "CTorchSource.hh"
#include "CGunSource.hh"

// optickscore-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"

// npy-
#include "Timer.hpp"
#include "TorchStepNPY.hpp"
#include "NGunConfig.hpp"
#include "GLMFormat.hpp"
#include "NLog.hpp"

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
    m_opticks->Summary("CG4::init opticks summary");
    TIMER("init");
}

void CG4::configure()
{
    configureEngine(); 


   /*
    m_cfg = m_opticks->getCfg();  
    m_cfg->commandline(argc, argv);   // why is this config deferred ... 
    m_cfg->dump(); 
    m_cache = new GCache(m_opticks);
    */

    //initEvent();

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


/*
void CG4::initEvent()
{
    // TODO: move evt setup outside CG4
    m_evt = m_opticks->makeEvent();  

    NPY<float>* nopstep = NPY<float>::make(0,4,4) ;  
    m_evt->setNopstepData(nopstep);

    assert(m_evt->getNumPhotons() == 0);
    assert(m_evt->getNumRecords() == 0);

    m_evt->createBuffers();  // the zeros means they are dynamic buffers

    // maybe can just make ni=0 "dynamic" buffers by default ? within the Opticks::makeEvt ? 
}
*/



void CG4::execute(const char* path)
{
    if(path && strlen(path) < 3) 
    {
        LOG(info) << "CG4::execute skip short path [" << path << "]" ;
        return ; 
    } 

    std::string cmd("/control/execute ");
    cmd += path ; 
    LOG(info) << "CG4::execute [" << cmd << "]" ; 
    m_uiManager->ApplyCommand(cmd);
}


void CG4::initialize()
{
    LOG(info) << "CG4::initialize" ;

    m_runManager->SetUserInitialization(new ActionInitialization(m_pga, m_sa)) ;
    m_runManager->Initialize();

#ifdef OLDPHYS
#else
    m_physics->setProcessVerbosity(0); 
#endif

    TIMER("initialize");

    postinitialize();
}

void CG4::postinitialize()
{
    setupCompressionDomains(); // should this be at end of configureDetector after the traverse has established domains ?


    m_uiManager = G4UImanager::GetUIpointer();

    std::string inimac = m_cfg->getG4IniMac();
    if(!inimac.empty()) execute(inimac.c_str()) ;

    LOG(info) << "CG4::postinitialize DONE" ;
    TIMER("postinitialize");
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
              ; 
    TIMER("_propagate");

    m_runManager->BeamOn(num_g4event);

    std::string runmac = m_cfg->getG4RunMac();
    LOG(info) << "CG4::propagate [" << runmac << "]"  ;
    if(!runmac.empty()) execute(runmac.c_str());

    TIMER("propagate");

    postpropagate();
}

void CG4::postpropagate()
{
    std::string finmac = m_cfg->getG4FinMac();
    LOG(info) << "CG4::postpropagate [" << finmac << "]"  ;
    if(!finmac.empty()) execute(finmac.c_str());


    m_evt->indexPhotonsCPU();
}



void CG4::save()
{
    m_evt->save(true);
}


void CG4::configurePhysics()
{
#ifdef OLDPHYS    
    LOG(info) << "CG4::configurePhysics old PhysicsList" ; 
    m_physics = new PhysicsList();
#else
    LOG(info) << "CG4::configurePhysics OpNovicePhysicsList" ; 
    m_physics = new OpNovicePhysicsList();
#endif
    // processes instanciated only after PhysicsList Construct that happens at runInitialization 

    m_runManager->SetUserInitialization(m_physics);
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
        OpticksQuery* query = NULL ;  // normally no OPTICKS_QUERY geometry subselection with test geometries
        detector  = static_cast<CDetector*>(new CTestDetector(m_cache, ggtc, query)) ; 
    }
    else
    {
        // no options here: will load the .gdml sidecar of the geocache .dae 
        LOG(info) << "CG4::configureDetector G4 GDML geometry " ; 
        OpticksQuery* query = m_cache->getQuery();
        detector  = static_cast<CDetector*>(new CGDMLDetector(m_cache, query)) ; 
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
        NGunConfig* gc = new NGunConfig();
        gc->parse(m_cfg->getG4GunConfig());

        unsigned int frameIndex = gc->getFrame() ;
        unsigned int numTransforms = m_detector->getNumGlobalTransforms() ;

        if(frameIndex < numTransforms )
        {
             const char* pvname = m_detector->getPVName(frameIndex);
             LOG(info) << "CG4::configureGenerator G4GUN"
                       << " frameIndex " << frameIndex 
                       << " numTransforms " << numTransforms 
                       << " pvname " << pvname 
                       ;

             glm::mat4 frame = m_detector->getGlobalTransform( frameIndex );
             gc->setFrameTransform(frame) ;
        }
        else
        {
             LOG(warning) << "CG4::configureGenerator gun config frameIndex not in detector"
                          << " frameIndex " << frameIndex
                          << " numTransforms " << numTransforms
                          ;
        }  

        m_evt->setNumG4Event(gc->getNumber()); 
        m_evt->setNumPhotonsPerG4Event(0); 

        int g4gun_verbosity = m_cfg->hasOpt("g4gundbg") ? 10 : 0 ; 
        CGunSource* gun = new CGunSource(g4gun_verbosity) ;
        gun->configure(gc);      

        source  = static_cast<CSource*>(gun); 
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
    m_recorder = new CRecorder(m_lib, m_evt, stepping_verbosity ); 
    m_rec = new Rec(m_lib, m_evt) ; 
    if(m_cfg->hasOpt("primary"))
         m_recorder->setupPrimaryRecording();

    source->setRecorder(m_recorder);

    m_pga = new CPrimaryGeneratorAction(source) ;
    TIMER("configureGenerator");
}



void CG4::configureStepping()
{
    m_steprec = new CStepRec(m_evt) ;  
    m_sa = new CSteppingAction(this) ;
    TIMER("configureStepping");
}

void CG4::setupCompressionDomains()
{
    m_detector->dumpPV("CG4::setupCompressionDomains dumpPV");

    glm::vec4 ce = m_detector->getCenterExtent();

    LOG(info) << "CG4::setupCompressionDomains"
              << " center_extent " << gformat(ce) 
              ;    


    m_opticks->setSpaceDomain(ce);

    m_evt->setTimeDomain(m_opticks->getTimeDomain());  
    m_evt->setWavelengthDomain(m_opticks->getWavelengthDomain()) ; 
    m_evt->setSpaceDomain(m_opticks->getSpaceDomain());

    m_evt->dumpDomains("CG4::setupCompressionDomains");
}


void CG4::cleanup()
{
    LOG(info) << "CG4::cleanup opening geometry" ; 
    G4GeometryManager::GetInstance()->OpenGeometry();
}



CG4::~CG4()
{
    //delete m_detector ; 
    //delete m_runManager;
}



