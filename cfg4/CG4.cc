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
#include "CGenerator.hh"


#include "CRecorder.hh"
#include "Rec.hh"
#include "CStepRec.hh"

#include "CPrimaryGeneratorAction.hh"
#include "CSteppingAction.hh"
#include "CRunAction.hh"
#include "CEventAction.hh"


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
CDetector* CG4::getDetector()
{
    return m_detector ; 
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
     m_generator(new CGenerator(m_hub, this)),
     m_material_table(NULL),
     m_recorder(new CRecorder(m_hub, m_lib, m_generator->isDynamic())), 
     m_rec(new Rec(m_hub, m_lib, m_generator->isDynamic())), 
     m_steprec(new CStepRec(m_hub, m_generator->isDynamic())),  
     m_collector(NULL),
     m_visManager(NULL),
     m_uiManager(NULL),
     m_ui(NULL),
     m_pga(new CPrimaryGeneratorAction(m_generator->getSource())),
     m_sa(new CSteppingAction(this, m_generator->isDynamic())),
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

void CG4::configure()
{
    LOG(info) << "CG4::configure" ;
    TIMER("configure");
}


void CG4::setUserInitialization(G4VUserDetectorConstruction* detector)
{
    m_runManager->SetUserInitialization(detector);
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
    TIMER("initialize");
    postinitialize();
}

void CG4::postinitialize()
{
    m_uiManager = G4UImanager::GetUIpointer();

    assert(m_cfg);    

    std::string inimac = m_cfg->getG4IniMac();
    if(!inimac.empty()) execute(inimac.c_str()) ;


    m_physics->setProcessVerbosity(0); 

    // needs to be after the detector Construct creates the materials
    m_material_table = new CMaterialTable(m_ok->getMaterialPrefix());
    //m_material_table->dump("CG4::postinitialize");

    LOG(info) << "CG4::postinitialize DONE" ;
    TIMER("postinitialize");
}


std::map<std::string, unsigned>& CG4::getMaterialMap()
{
    assert(m_material_table);
    return m_material_table->getMaterialMap();
}

void CG4::interactive()
{
    bool g4ui = m_ok->hasOpt("g4ui") ; 
    if(!g4ui) return ; 

    LOG(info) << "CG4::interactive proceeding " ; 

    m_visManager = new G4VisExecutive;
    m_visManager->Initialize();

    m_ui = new G4UIExecutive(m_ok->getArgc(), m_ok->getArgv());
    m_ui->SessionStart();
}


OpticksEvent* CG4::initEvent()
{
    OpticksEvent* evt = m_hub->createG4Event();   
    assert(evt->isG4());
    evt->dumpDomains("CG4::propagate");
 
    unsigned int numG4Evt = m_generator->getNumG4Event();

    NPY<float>* gs = m_generator->getGensteps(); 
    if(gs)
    {
        // WHEN OPERATING FROM GENSTEP (EG WITH TORCH SOURCE) 
        // YOU KNOW THE TOTAL NUMBER OF PHOTONS AHEAD OF TIME
        // WHICH ALLOWS STATIC RUNNING 

        evt->setGenstepData(gs);   // <-- this will switch on static running 
        evt->zero();               // no merit in deferring allocation CPU side
    } 
    else
    {
         LOG(info) << "CG4::initEvent"
                   << " no genstep (DYNAMIC RUNNING) "
                   ;  
    }

    evt->setNumG4Event(numG4Evt) ; 
    evt->setNumPhotonsPerG4Event(m_generator->getNumPhotonsPerG4Event()) ; 


    m_recorder->initEvent();
    m_rec->initEvent();

    return evt ; 
}

void CG4::propagate()
{
    OpticksEvent* evt = initEvent();
    unsigned int numG4Evt = evt->getNumG4Event();

    LOG(info) << "CG4::propagate" << " numG4Evt " << numG4Evt ; 

    TIMER("_propagate");

    m_runManager->BeamOn(numG4Evt);

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

    m_collector = OpticksG4Collector::Instance() ;
    m_collector->Summary("CG4::postpropagate");

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

void CG4::cleanup()
{
    LOG(info) << "CG4::cleanup opening geometry" ; 
    G4GeometryManager::GetInstance()->OpenGeometry();
}

