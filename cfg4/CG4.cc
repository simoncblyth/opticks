// cfg4-;cfg4--;ggv-;ggv-pmt-test --cfg4 
// cfg4-;cfg4--;op --cfg4 --g4gun --dbg 
// cfg4-;cfg4--;ggv-;ggv-g4gun --dbg

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"

// opticksgeo-
#include "OpticksHub.hh"

// npy-
#include "NPY.hpp"
#include "Timer.hpp"
#include "TorchStepNPY.hpp"
#include "NGunConfig.hpp"
#include "GLMFormat.hpp"

//ggeo-
#include "GBndLib.hh"
#include "GGeoTestConfig.hh"


#include "CFG4_PUSH.hh"
//g4-
#include "G4RunManager.hh"
#include "G4String.hh"

#include "G4VisExecutive.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4GeometryManager.hh"


//cg4-
#include "ActionInitialization.hh"

#ifdef OLDPHYS
#include "PhysicsList.hh"
#else
#include "OpNovicePhysicsList.hh"
#endif

#include "OpticksG4Collector.hh"


#include "CTestDetector.hh"
#include "CGDMLDetector.hh"
#include "CPropLib.hh"
#include "CRecorder.hh"
#include "Rec.hh"
#include "CStepRec.hh"

#include "CPrimaryGeneratorAction.hh"
#include "CSteppingAction.hh"
#include "CRunAction.hh"
#include "CEventAction.hh"

#include "CTorchSource.hh"
#include "CGunSource.hh"
#include "CMaterialTable.hh"
#include "CG4.hh"


#include "CFG4_POP.hh"
#include "CFG4_BODY.hh"




#include "PLOG.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



CG4::CG4(OpticksHub* hub) 
   :
     OpticksEngine(hub),
     m_torch(NULL),
     m_detector(NULL),
     m_material_table(NULL),
     m_lib(NULL),
     m_recorder(NULL),
     m_rec(NULL),
     m_steprec(NULL),
     m_collector(NULL),
     m_physics(NULL),
     m_runManager(NULL),
     m_g4ui(false),
     m_visManager(NULL),
     m_uiManager(NULL),
     m_ui(NULL),
     m_pga(NULL),
     m_sa(NULL),
     m_ra(NULL),
     m_ea(NULL)
{
     init();
}


CRecorder* CG4::getRecorder()
{
    return m_recorder ; 
}
Rec* CG4::getRec()
{
    return m_rec ; 
}
CStepRec* CG4::getStepRec()
{
    return m_steprec ; 
}
CPropLib* CG4::getPropLib()
{
    return m_lib ; 
}


void CG4::init()
{
    m_opticks->Summary("CG4::init opticks summary");
    TIMER("init");
}

void CG4::configure()
{
    m_g4ui = m_opticks->hasOpt("g4ui") ; 
    LOG(info) << "CG4::configure"
              << " g4ui " << m_g4ui
              ; 

    m_runManager = new G4RunManager;

    configurePhysics();
    configureDetector();

    glm::vec4 ce = m_detector->getCenterExtent();
    LOG(info) << "CG4::configure"
              << " center_extent " << gformat(ce) 
              ;    

    m_opticks->setSpaceDomain(ce); // triggers Opticks::configureDomains
    OpticksEvent* evt = m_hub->createEvent();   // HMM: feels too soon, when thinking multi-event, remember not 1-to-1 between Opticks events and G4  
    evt->dumpDomains("CG4::configure");

    configureGenerator();
    configureStepping();

    m_ra = new CRunAction ; 
    m_ea = new CEventAction ;  

    TIMER("configure");
}


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

    m_runManager->SetUserInitialization(new ActionInitialization(m_pga, m_sa, m_ra, m_ea)) ;
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
    m_uiManager = G4UImanager::GetUIpointer();

    assert(m_cfg);    

    std::string inimac = m_cfg->getG4IniMac();
    if(!inimac.empty()) execute(inimac.c_str()) ;


    // needs to be after the detector Construct creates the materials
    m_material_table = new CMaterialTable(m_opticks->getMaterialPrefix());
    m_material_table->dump("CG4::postinitialize");


    LOG(info) << "CG4::postinitialize DONE" ;
    TIMER("postinitialize");
}


std::map<std::string, unsigned>& CG4::getMaterialMap()
{
    assert(m_material_table);
    return m_material_table->getMaterialMap();
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
    OpticksEvent* evt = m_hub->getEvent();
    assert(evt);
    unsigned int num_g4event = evt->getNumG4Event();
 
    LOG(info) << "CG4::propagate"
              << " num_g4event " << evt->getNumG4Event()
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

    // G4 specific things belongs here

    m_collector = OpticksG4Collector::Instance() ;
    m_collector->Summary("CG4::postpropagate");

    //NPY<float>* gs = m_collector->getGensteps();
    //gs->save("$TMP/CG4Test_genstep.npy");

    OpticksEvent* evt = m_hub->getEvent();
    assert(evt);
    evt->postPropagateGeant4();
}

NPY<float>* CG4::getGensteps()
{
    m_collector = OpticksG4Collector::Instance();
    NPY<float>* gs = m_collector->getGensteps();
    // need some kind of reset ready for next event
    return gs ; 
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
    if(m_opticks->hasOpt("test"))
    {
        LOG(info) << "CG4::configureDetector G4 simple test geometry " ; 
        std::string testconfig = m_cfg->getTestConfig();
        GGeoTestConfig* ggtc = new GGeoTestConfig( testconfig.empty() ? NULL : testconfig.c_str() );
        OpticksQuery* query = NULL ;  // normally no OPTICKS_QUERY geometry subselection with test geometries
        detector  = static_cast<CDetector*>(new CTestDetector(m_opticks, ggtc, query)) ; 
    }
    else
    {
        // no options here: will load the .gdml sidecar of the geocache .dae 
        LOG(info) << "CG4::configureDetector G4 GDML geometry " ; 
        OpticksQuery* query = m_opticks->getQuery();
        detector  = static_cast<CDetector*>(new CGDMLDetector(m_opticks, query)) ; 
    }

    m_detector = detector ; 
    m_lib = detector->getPropLib();
    m_runManager->SetUserInitialization(detector);
    TIMER("configureDetector");
}


void CG4::configureGenerator()
{
    CSource* source = NULL ; 

    // THIS IS AN EVENT LEVEL THING : RENAME initEvent ?
    // HMM THIS CODE LOOKS TO BE DUPLICITOUS AND OUT OF PLACE : NEEDS MOVING 


    OpticksEvent* evt = m_hub->getEvent();
    assert(evt);

    if(m_opticks->getSourceCode() == TORCH)
    {
        LOG(info) << "CG4::configureGenerator TORCH " ; 
        m_torch = m_opticks->makeSimpleTorchStep();
        m_torch->addStep(true); // calls update setting pos,dir,pol using the frame transform and preps the NPY buffer
        m_torch->Summary("CG4::configure TorchStepNPY::Summary");

        evt->setGenstepData( m_torch->getNPY() );  // sets the number of photons and preps buffers (unallocated)
        evt->setNumG4Event(m_torch->getNumG4Event()); 
        evt->setNumPhotonsPerG4Event(m_torch->getNumPhotonsPerG4Event()); 

        evt->zero();  
        // IS THIS ALWAYS NEEDED WITH setGenstepData
        // OPERATING FROM GENSTEP : YOU KNOW THE TOTAL NUMBER OF PHOTONS AHEAD OF TIME


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

        evt->setNumG4Event(gc->getNumber()); 
        evt->setNumPhotonsPerG4Event(0); 

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
    m_recorder = new CRecorder(m_lib, evt, stepping_verbosity ); 

    m_rec = new Rec(m_lib, evt) ; 

    //if(m_cfg->hasOpt("primary"))
    //     m_recorder->setupPrimaryRecording();

    source->setRecorder(m_recorder);

    m_pga = new CPrimaryGeneratorAction(source) ;
    TIMER("configureGenerator");
}



void CG4::configureStepping()
{
    m_steprec = new CStepRec(m_hub) ;  
    m_sa = new CSteppingAction(this) ;
    TIMER("configureStepping");
}


void CG4::cleanup()
{
    LOG(info) << "CG4::cleanup opening geometry" ; 
    G4GeometryManager::GetInstance()->OpenGeometry();
}



