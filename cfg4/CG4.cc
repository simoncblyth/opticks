// cfg4-;cfg4--;ggv-;ggv-pmt-test --cfg4 
// cfg4-;cfg4--;op --cfg4 --g4gun --dbg 
// cfg4-;cfg4--;ggv-;ggv-g4gun --dbg

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"

// opticksgeo-
#include "OpticksHub.hh"
#include "OpticksRun.hh"

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

#include "CRandomEngine.hh"
#include "CPhysics.hh"
#include "CGeometry.hh"
#include "CMaterialLib.hh"
#include "CDetector.hh"
#include "CGenerator.hh"

#include "CCollector.hh"
#include "CRecorder.hh"
#include "CStepRec.hh"

#include "CPrimaryGeneratorAction.hh"
#include "CSteppingAction.hh"
#include "CTrackingAction.hh"
#include "CRunAction.hh"
#include "CEventAction.hh"


#include "CRayTracer.hh"
#include "CMaterialTable.hh"
#include "CG4.hh"


#include "CFG4_POP.hh"
#include "CFG4_BODY.hh"

#include "PLOG.hh"



Opticks* CG4::getOpticks()
{
    return m_ok ; 
}
OpticksHub* CG4::getHub()
{
    return m_hub ; 
}
CGeometry* CG4::getGeometry()
{
    return m_geometry ; 
}
CMaterialBridge* CG4::getMaterialBridge()
{
    return m_geometry->getMaterialBridge() ; 
}
CSurfaceBridge* CG4::getSurfaceBridge()
{
    return m_geometry->getSurfaceBridge() ; 
}
CRandomEngine* CG4::getRandomEngine() const 
{
    return m_engine ; 
}



CRecorder* CG4::getRecorder()
{
    return m_recorder ; 
}

CStepRec* CG4::getStepRec()
{
    return m_steprec ; 
}


CMaterialLib* CG4::getMaterialLib()
{
    return m_mlib ; 
}


CDetector* CG4::getDetector()
{
    return m_detector ; 
}


CG4Ctx& CG4::getCtx()
{
    return m_ctx ; 
}

double CG4::getCtxRecordFraction() const 
{
    return m_ctx._record_fraction ; 
}

unsigned long long CG4::getSeqHis() const 
{
    return m_recorder->getSeqHis() ;
}
unsigned long long CG4::getSeqMat() const 
{
    return m_recorder->getSeqMat() ; 
}


const CG4* CG4::INSTANCE = NULL ; 

CG4::CG4(OpticksHub* hub) 
   :
     m_hub(hub),
     m_ok(m_hub->getOpticks()),
     m_run(m_ok->getRun()),
     m_cfg(m_ok->getCfg()),
     m_ctx(m_ok),
     m_engine(m_ok->isAlign() ? new CRandomEngine(this) : NULL ),
     m_physics(new CPhysics(this)),
     m_runManager(m_physics->getRunManager()),
     m_geometry(new CGeometry(m_hub)),
     m_hookup(m_geometry->hookup(this)),
     m_mlib(m_geometry->getMaterialLib()),
     m_detector(m_geometry->getDetector()),
     m_generator(new CGenerator(m_hub, this)),
     m_dynamic(m_generator->isDynamic()),
     m_collector(NULL),   // deferred instanciation until CG4::postinitialize after G4 materials have overridden lookupA
     m_recorder(new CRecorder(this, m_geometry, m_dynamic)), 
     m_steprec(new CStepRec(m_ok, m_dynamic)),  
     m_visManager(NULL),
     m_uiManager(NULL),
     m_ui(NULL),
     m_pga(new CPrimaryGeneratorAction(m_generator->getSource())),
     m_sa(new CSteppingAction(this, m_generator->isDynamic())),
     m_ta(new CTrackingAction(this)),
     m_ra(new CRunAction(m_hub)),
     m_ea(new CEventAction(this)),
     m_rt(new CRayTracer(this)),
     m_initialized(false)
{
     OK_PROFILE("CG4::CG4");
     init();
     INSTANCE = this ; 
}

void CG4::init()
{
    //m_ok->Summary("CG4::init opticks summary");

    LOG(info) << "CG4::init"  << " ctx " << m_ctx.desc() ; 
     
    initialize();

}


void CG4::setUserInitialization(G4VUserDetectorConstruction* detector)
{
    m_runManager->SetUserInitialization(detector);
}


void CG4::initialize()
{
    assert(!m_initialized && "CG4::initialize already initialized");
    m_initialized = true ; 
    LOG(info) << "CG4::initialize" ;


    m_runManager->SetUserAction(m_ra);
    m_runManager->SetUserAction(m_ea);
    m_runManager->SetUserAction(m_pga);
    m_runManager->SetUserAction(m_ta);
    m_runManager->SetUserAction(m_sa);


    m_runManager->Initialize();
    postinitialize();
}



CEventAction* CG4::getEventAction()
{
    return m_ea ;    
}
CTrackingAction* CG4::getTrackingAction()
{
    return m_ta ;    
}
CSteppingAction* CG4::getSteppingAction()
{
    return m_sa ;    
}


//int CG4::getStepId()
//{
//    return m_sa->getStepId();
//}



void CG4::postinitialize()
{
    m_uiManager = G4UImanager::GetUIpointer();

    assert(m_cfg);    

    std::string inimac = m_cfg->getG4IniMac();
    if(!inimac.empty()) execute(inimac.c_str()) ;

    m_physics->setProcessVerbosity(0); 

    // needs to be after the detector Construct creates the materials

    // postinitialize order matters, creates/shares m_material_bridge instance

    m_geometry->postinitialize();
    m_recorder->postinitialize();  
    //m_rec->postinitialize();


    m_ea->postinitialize();

    m_ta->postinitialize();

    m_sa->postinitialize();



    m_hub->overrideMaterialMapA( getMaterialMap(), "CG4::postinitialize/g4mm") ;  // for translation of material indices into GPU texture lines 

    m_collector = new CCollector(m_hub) ; // currently hub just used for material code lookup, not evt access


    if(m_ok->isG4Snap()) snap() ;



    LOG(info) << "CG4::postinitialize DONE" ;
}


std::map<std::string, unsigned>& CG4::getMaterialMap()
{
    return m_geometry->getMaterialMap();
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



void CG4::snap()
{
    m_visManager = new G4VisExecutive;
    m_visManager->Initialize();
    m_rt->snap();
}


void CG4::posttrack()
{
    if(m_ctx._optical)
    {
        m_recorder->posttrack();
    } 
    if(m_engine)
    {
        m_engine->posttrack();
    }
}

void CG4::poststep()
{
    if(m_engine)
    {
        m_engine->poststep();
    }
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


void CG4::initEvent(OpticksEvent* evt)
{
    m_generator->configureEvent(evt);

    m_ctx.initEvent(evt);

    m_recorder->initEvent(evt);

    NPY<float>* nopstep = evt->getNopstepData();
    if(!nopstep) LOG(fatal) << " nopstep NULL " << " evt " << evt->getShapeString() ; 
    assert(nopstep); 
    m_steprec->initEvent(nopstep);
}



NPY<float>* CG4::propagate()
{
    OpticksEvent* evt = m_run->getG4Event();
    LOG(info) << evt->brief() <<  " " << evt->getShapeString() ;


    bool isg4evt = evt && evt->isG4() ;  

    if(!isg4evt)
        LOG(fatal) << "CG4::propagate expecting G4 Opticks Evt " ; 

    assert(isg4evt);

    if(m_ok->isFabricatedGensteps())
    {
        bool hasGensteps = evt->hasGenstepData();
        if(!hasGensteps) 
             LOG(fatal) << "MUST: OpticksRun::setGensteps before CG4::propagate when ok.isFabricatedGensteps " ; 
        assert(hasGensteps);    
    }

    LOG(info) << "CG4::propagate(" << m_ok->getTagOffset() << ") " << evt->getDir() ; 

    initEvent(evt);

    unsigned int numG4Evt = evt->getNumG4Event();

    OK_PROFILE("_CG4::propagate");

    m_runManager->BeamOn(numG4Evt);

    OK_PROFILE("CG4::propagate");

    std::string runmac = m_cfg->getG4RunMac();
    if(!runmac.empty()) execute(runmac.c_str());

    postpropagate();

    NPY<float>* gs = m_collector->getGensteps(); 

    NPY<float>* pr = m_collector->getPrimary(); 
    pr->save("$TMP/cg4/primary.npy");   // debugging primary position issue 

    return gs ; 
}

void CG4::postpropagate()
{
    LOG(info) << "CG4::postpropagate(" << m_ok->getTagOffset() << ")"
              << " ctx " << m_ctx.desc_stats() 
               ;

    std::string finmac = m_cfg->getG4FinMac();
    if(!finmac.empty()) execute(finmac.c_str());

    OpticksEvent* evt = m_run->getG4Event();
    assert(evt);
    evt->postPropagateGeant4();


    dynamic_cast<CSteppingAction*>(m_sa)->report("CG4::postpropagate");



    if(m_engine) m_engine->postpropagate();  

    LOG(info) << "CG4::postpropagate(" << m_ok->getTagOffset() << ") DONE"  ;
}


NPY<float>* CG4::getGensteps()
{
    return m_collector->getGensteps();
}


void CG4::cleanup()
{
    LOG(info) << "CG4::cleanup opening geometry" ; 
    G4GeometryManager::GetInstance()->OpenGeometry();
}

