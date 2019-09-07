/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <csignal>
#include "CFG4_BODY.hh"

#include "SLog.hh"

// okc-
#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"
#include "OpticksPhoton.h"
#include "OpticksCfg.hh"

// okg-
#include "OpticksHub.hh"
#include "OpticksRun.hh"

// npy-
#include "NPY.hpp"
#include "NLookup.hpp"

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

//cfg4-
#include "CRandomEngine.hh"
#include "CMixMaxRng.hh"

#include "CPhysics.hh"
#include "CGeometry.hh"
#include "CMaterialLib.hh"
#include "CDetector.hh"
#include "CGenerator.hh"
#include "CGenstepCollector.hh"
#include "CPrimaryCollector.hh"
#include "CRecorder.hh"
#include "CStepRec.hh"

#include "CPrimaryGeneratorAction.hh"
#include "CSteppingAction.hh"
#include "CTrackingAction.hh"
#include "CRunAction.hh"
#include "CEventAction.hh"
#include "CSensitiveDetector.hh"

#include "CRayTracer.hh"
#include "CMaterialTable.hh"
#include "CG4.hh"
#include "C4FPEDetection.hh"

#include "CFG4_POP.hh"
#include "CFG4_BODY.hh"

#include "PLOG.hh"

const plog::Severity CG4::LEVEL = PLOG::EnvLevel("CG4", "DEBUG") ; 


CG4* CG4::INSTANCE = NULL ; 

Opticks*    CG4::getOpticks() const { return m_ok ; } 
OpticksHub* CG4::getHub() const { return m_hub ; }
OpticksRun* CG4::getRun() const { return m_run ; } 

CRandomEngine*     CG4::getRandomEngine() const { return m_engine ; }
CGenerator*        CG4::getGenerator() const { return m_generator ; }
CRecorder*         CG4::getRecorder() const { return m_recorder ; }
unsigned long long CG4::getSeqHis() const { return m_recorder->getSeqHis() ; }
unsigned long long CG4::getSeqMat() const { return m_recorder->getSeqMat() ; }
CStepRec*          CG4::getStepRec() const { return m_steprec ; }
CGeometry*         CG4::getGeometry() const { return m_geometry ; }
CMaterialBridge*   CG4::getMaterialBridge() const { return m_geometry->getMaterialBridge() ; }


CSurfaceBridge*    CG4::getSurfaceBridge() const { return m_geometry->getSurfaceBridge() ; }
CMaterialLib*      CG4::getMaterialLib() const { return m_mlib ; }
CDetector*         CG4::getDetector() const  { return m_detector ; }
CEventAction*      CG4::getEventAction() const { return m_ea ;    }
CTrackingAction*   CG4::getTrackingAction() const { return m_ta ;    }
CSensitiveDetector* CG4::getSensitiveDetector() const { return m_sd ;    }
CSteppingAction*   CG4::getSteppingAction() const { return m_sa ;    } 

double             CG4::getCtxRecordFraction() const { return m_ctx._record_fraction ; }
NPY<float>*        CG4::getGensteps() const { return m_collector->getGensteps(); } 

const std::map<std::string, unsigned>& CG4::getMaterialMap() const 
{
    return m_geometry->getMaterialMap();
}
double CG4::flat_instrumented(const char* file, int line)
{
    return m_engine ? m_engine->flat_instrumented(file, line) : G4UniformRand() ; 
}
CG4Ctx& CG4::getCtx()
{
    return m_ctx ; 
}


int CG4::preinit()
{
    OK_PROFILE("_CG4::CG4");
    if(m_ok->hasOpt("cg4sigint")) std::raise(SIGINT);   // <-- handy location for setting G4 breakpoints
    return 0 ; 
}

CG4::CG4(OpticksHub* hub) 
    :
    m_log(new SLog("CG4::CG4", "", LEVEL)),
    m_hub(hub),
    m_ok(m_hub->getOpticks()),
    m_preinit(preinit()),
    m_run(m_ok->getRun()),
    m_cfg(m_ok->getCfg()),
    m_ctx(m_ok),
    m_engine(m_ok->isAlign() ? new CRandomEngine(this) : NULL  ),   // --align
    m_physics(new CPhysics(this)),
    m_runManager(m_physics->getRunManager()),
    m_sd(new CSensitiveDetector("SD0")),
    m_geometry(new CGeometry(m_hub, m_sd)),
    m_hookup(m_geometry->hookup(this)),
    m_mlib(m_geometry->getMaterialLib()),
    m_detector(m_geometry->getDetector()),
    m_generator(new CGenerator(m_hub->getGen(), this)),
    m_dynamic(m_generator->isDynamic()),
    m_collector(NULL),   // deferred instanciation until CG4::postinitialize after G4 materials have overridden lookupA
    m_primary_collector(new CPrimaryCollector),
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
    (*m_log)("DONE");
}


void CG4::init()
{
    LOG(LEVEL) << " ctx " << m_ctx.desc() ; 
    initialize();
}

/**
CG4::setUserInitialization
---------------------------

Invoked from above initializer list(m_hookup) via CGeometry::hookup 
after CGeometry/m_geometry is initialized.

**/
void CG4::setUserInitialization(G4VUserDetectorConstruction* detector)
{
    m_runManager->SetUserInitialization(detector);
}

void CG4::initialize()
{
    LOG(LEVEL) << "[" ;
    assert(!m_initialized && "CG4::initialize already initialized");
    m_initialized = true ; 



    m_runManager->SetUserAction(m_ra);
    m_runManager->SetUserAction(m_ea);
    m_runManager->SetUserAction(m_pga);
    m_runManager->SetUserAction(m_ta);
    m_runManager->SetUserAction(m_sa);

    m_runManager->Initialize();
    postinitialize();
    LOG(LEVEL) << "]" ;
}

void CG4::postinitialize()
{
    LOG(LEVEL) << "[" ;
    C4FPEDetection::InvalidOperationDetection_Disable();  // see notes/issues/OKG4Test_prelaunch_FPE_causing_fail.rst

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

    // for translation of material indices into GPU texture lines 
    m_hub->overrideMaterialMapA( getMaterialMap(), "CG4::postinitialize/g4mm") ; 

    NLookup* lookup = m_hub->getLookup();
    lookup->close() ;  // hmm what about B (from GBndLiB)  

    m_collector = new CGenstepCollector(lookup) ; 

    if(m_ok->isG4Snap()) snap() ;

    LOG(LEVEL) << "]" ;
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

// invoked from CTrackingAction::PreUserTrackingAction immediately after CG4Ctx::setTrack
void CG4::preTrack()
{
    if(m_engine)
    {
        m_engine->preTrack();
    }
}

// invoked from CTrackingAction::PostUserTrackingAction for optical photons only 
void CG4::postTrack()
{
    if(m_ctx._optical)
    {
        m_recorder->postTrack();
    } 
    if(m_engine)
    {
        m_engine->postTrack();
    }
}

// invoked from CSteppingAction::UserSteppingAction
void CG4::postStep()
{
    if(m_engine)
    {
        m_engine->postStep();
    }
}

void CG4::interactive()
{
    bool g4ui = m_ok->hasOpt("g4ui") ; 
    if(!g4ui) return ; 

    m_visManager = new G4VisExecutive;
    m_visManager->Initialize();

    m_ui = new G4UIExecutive(m_ok->getArgc(), m_ok->getArgv());
    m_ui->SessionStart();
}

void CG4::initEvent(OpticksEvent* evt)
{
    LOG(LEVEL) << "[" ;
    m_generator->configureEvent(evt);

    m_ctx.initEvent(evt);

    m_recorder->initEvent(evt);

    NPY<float>* nopstep = evt->getNopstepData();
    if(!nopstep) LOG(fatal) << " nopstep NULL " << " evt " << evt->getShapeString() ; 
    assert(nopstep); 
    m_steprec->initEvent(nopstep);
    LOG(LEVEL) << "]" ;
}


NPY<float>* CG4::propagate()
{
    LOG(LEVEL) << "[" ;
    OpticksEvent* evt = m_run->getG4Event();
    LOG(LEVEL) << evt->brief() <<  " " << evt->getShapeString() ;

    bool isg4evt = evt && evt->isG4() ;  

    if(!isg4evt)
        LOG(fatal) << "CG4::propagate expecting G4 Opticks Evt " ; 

    assert(isg4evt);

    //if(m_ok->isFabricatedGensteps())
    if(m_generator->hasGensteps())
    {
        bool hasGensteps = evt->hasGenstepData();
        if(!hasGensteps) 
             LOG(fatal) << "OpticksRun CGenerator gensteps inconsistency " ; 
        assert(hasGensteps);    
    }

    LOG(LEVEL) << "(" << m_ok->getTagOffset() << ") " << evt->getDir() ; 

    initEvent(evt);

    unsigned int numG4Evt = evt->getNumG4Event();


    LOG(LEVEL) << " calling BeamOn numG4Evt " << numG4Evt ; 
    OK_PROFILE("_CG4::propagate");

    m_runManager->BeamOn(numG4Evt);

    OK_PROFILE("CG4::propagate");
    LOG(LEVEL) << " calling BeamOn numG4Evt " << numG4Evt << " DONE " ; 

    std::string runmac = m_cfg->getG4RunMac();
    if(!runmac.empty()) execute(runmac.c_str());

    postpropagate();

    NPY<float>* gs = m_collector->getGensteps(); 


    LOG(LEVEL) << "idpath " << m_ok->getIdPath();  

    //NPY<float>* pr = m_primary_collector->getPrimary(); 
    //pr->save("$TMP/cg4/primary.npy");   // debugging primary position issue 

    LOG(LEVEL) << "]" ;
    return gs ; 
}

void CG4::postpropagate()
{
    LOG(info) 
         << "[" 
         << " (" << m_ok->getTagOffset() << ")"
         << " ctx " << m_ctx.desc_stats() 
         ;

    std::string finmac = m_cfg->getG4FinMac();
    if(!finmac.empty()) execute(finmac.c_str());

    OpticksEvent* evt = m_run->getG4Event();
    assert(evt);

    NPY<float>* so = m_generator->getSourcePhotons(); 
    if(so) evt->setSourceData(so); 

    evt->postPropagateGeant4();

    dynamic_cast<CSteppingAction*>(m_sa)->report("CG4::postpropagate");

    if(m_engine) m_engine->postpropagate();  

    LOG(info) << "]" 
              << " (" <<  m_ok->getTagOffset() << ")"  
              ;
}

void CG4::addRandomNote(const char* note, int value)
{
    assert( m_engine ); 
    m_engine->addNote(note, value); 
}


void CG4::cleanup()
{
    LOG(info) << "[" ; 
    G4GeometryManager::GetInstance()->OpenGeometry();
    LOG(info) << "]" ; 
}

