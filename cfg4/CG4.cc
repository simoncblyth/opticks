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


#include "CFG4_PUSH.hh"
//g4-
#include "G4String.hh"

#include "G4RunManager.hh"
#include "G4VisExecutive.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4GeometryManager.hh"


//cg4-
#include "ActionInitialization.hh"

#include "OpticksG4Collector.hh"


#include "CPhysics.hh"
#include "CGeometry.hh"
#include "CPropLib.hh"
#include "CDetector.hh"


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



CG4::CG4(OpticksHub* hub) 
   :
     m_hub(hub),
     m_ok(m_hub->getOpticks()),
     m_cfg(m_ok->getCfg()),
     m_physics(new CPhysics(m_hub)),
     m_runManager(m_physics->getRunManager()),
     m_geometry(new CGeometry(m_hub, this)),
     m_lib(m_geometry->getPropLib()),
     m_detector(m_geometry->getDetector()),
     m_torch(NULL),
     m_material_table(NULL),
     m_recorder(new CRecorder(m_hub, m_lib)), 
     m_rec(new Rec(m_hub, m_lib)), 
     m_steprec(new CStepRec(m_hub)),  
     m_collector(NULL),
     m_g4ui(false),
     m_visManager(NULL),
     m_uiManager(NULL),
     m_ui(NULL),
     m_pga(NULL),
     m_sa(new CSteppingAction(this)),
     m_ra(new CRunAction(m_hub)),
     m_ea(new CEventAction(m_hub)) 
{
     init();
}

void CG4::init()
{
    m_ok->Summary("CG4::init opticks summary");
    TIMER("init");
}

void CG4::setUserInitialization(G4VUserDetectorConstruction* detector)
{
    m_runManager->SetUserInitialization(detector);
}
 


void CG4::configure()
{
    m_g4ui = m_ok->hasOpt("g4ui") ; 
    LOG(info) << "CG4::configure"
              << " g4ui " << m_g4ui
              ; 

    glm::vec4 ce = m_detector->getCenterExtent();
    LOG(info) << "CG4::configure"
              << " center_extent " << gformat(ce) 
              ;    

    m_ok->setSpaceDomain(ce); // triggers Opticks::configureDomains

    // HMM: feels too soon, when thinking multi-event, 
    //      remember not 1-to-1 between Opticks events and G4  
    //      (non 1-to-1 is a kludge due to use of photons at top level of tree)
    //  
    OpticksEvent* evt = m_hub->createG4Event();   
    assert(evt->isG4());
    evt->dumpDomains("CG4::configure");

    configureGenerator();


    // actions canot be instanciated prior to physics setup 
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

    m_physics->setProcessVerbosity(0); 

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
    m_material_table = new CMaterialTable(m_ok->getMaterialPrefix());
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


void CG4::configureGenerator()
{
    CSource* source = NULL ; 

    // THIS IS AN EVENT LEVEL THING : RENAME initEvent ?
    // HMM THIS CODE LOOKS TO BE DUPLICITOUS AND OUT OF PLACE : NEEDS MOVING 

    OpticksEvent* evt = m_hub->getEvent();
    assert(evt);

    if(m_ok->getSourceCode() == TORCH)  // TORCH produces only optical photons
    {
        LOG(info) << "CG4::configureGenerator TORCH " ; 
        m_torch = m_ok->makeSimpleTorchStep();
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
    else if(m_ok->getSourceCode() == G4GUN)
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


        // TODO: avoid needing the evt at this early stage
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


    // recorder is back here in order to pass to source for primary recording (unused?)


    //if(m_cfg->hasOpt("primary"))
    //     m_recorder->setupPrimaryRecording();

    source->setRecorder(m_recorder);

    m_pga = new CPrimaryGeneratorAction(source) ;
    TIMER("configureGenerator");
}





void CG4::cleanup()
{
    LOG(info) << "CG4::cleanup opening geometry" ; 
    G4GeometryManager::GetInstance()->OpenGeometry();
}


