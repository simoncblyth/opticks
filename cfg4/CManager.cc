#include "Randomize.hh"
#include "Opticks.hh"
#include "OpticksEvent.hh"

#include "CTrack.hh"
#include "CRandomEngine.hh"
#include "CRecorder.hh"
#include "CStepRec.hh"
#include "CStepStatus.hh"
#include "CProcessManager.hh"
#include "CG4Ctx.hh"
#include "CManager.hh"


/*

// cg4-
#include "CBoundaryProcess.hh"

#include "G4Event.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "G4TransportationManager.hh"
#include "G4Navigator.hh"


#include "DsG4CompositeTrackInfo.h"
#include "DsPhotonTrackInfo.h"

#include "CStage.hh"
#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "CRecorder.hh"
#include "Format.hh"
#include "CPropLib.hh"
#include "CStp.hh"
#include "CSteppingAction.hh"
#include "CTrack.hh"
#include "CG4Ctx.hh"
#include "CG4.hh"

#include "CFG4_POP.hh"


// optickscore-
#include "Opticks.hh"
#include "OpticksFlags.hh"

*/




#include "PLOG.hh"

const plog::Severity CManager::LEVEL = PLOG::EnvLevel("CManager", "INFO"); 


CRecorder* CManager::getRecorder() const 
{ 
    return m_recorder ; 
}
CStepRec*  CManager::getStepRec() const 
{
    return m_steprec ; 
}



CRandomEngine* CManager::getRandomEngine() const 
{ 
    return m_engine ; 
}
CG4Ctx& CManager::getCtx()
{
    return *m_ctx ; 
}

double CManager::flat_instrumented(const char* file, int line)
{
    return m_engine ? m_engine->flat_instrumented(file, line) : G4UniformRand() ; 
}

unsigned long long CManager::getSeqHis() const { return m_recorder->getSeqHis() ; }



CManager::CManager(Opticks* ok, bool dynamic )
    :
    m_ok(ok),
    m_dynamic(dynamic),
    m_ctx(new CG4Ctx(m_ok)),
    m_engine(m_ok->isAlign() ? new CRandomEngine(this) : NULL  ),   // --align
    m_recorder(new CRecorder(*m_ctx, m_dynamic)),  // optical recording 
    m_steprec(new CStepRec(m_ok, m_dynamic)),      // non-optical recording 
    m_dbgflat(m_ok->isDbgFlat()), 
    m_dbgrec(m_ok->isDbgRec()),
    m_trman(NULL),
    m_nav(NULL),
    m_steprec_store_count(0), 
    m_cursor_at_clear(-1)
{
}



/*
   m_geometry(g4->getGeometry()),
   m_material_bridge(NULL),
   m_mlib(g4->getMaterialLib()),
   m_steprec(g4->getStepRec()),

*/



void CManager::setMaterialBridge(const CMaterialBridge* material_bridge)
{
    m_recorder->setMaterialBridge(material_bridge); 

    m_trman = G4TransportationManager::GetTransportationManager(); 
    m_nav = m_trman->GetNavigatorForTracking() ;
}


/**
CManager::initEvent
---------------------

Configure event recording, limits/shapes etc.. 

**/

void CManager::initEvent(OpticksEvent* evt)
{
    m_ctx->initEvent(evt);
    m_recorder->initEvent(evt);

    NPY<float>* nopstep = evt->getNopstepData();
    if(!nopstep) LOG(fatal) << " nopstep NULL " << " evt " << evt->getShapeString() ; 
    assert(nopstep); 
    m_steprec->initEvent(nopstep);
}

void CManager::BeginOfRunAction(const G4Run*)
{
    LOG(LEVEL); 
}
void CManager::EndOfRunAction(const G4Run*)
{
    LOG(LEVEL); 
}


/**
CManager::BeginOfEventAction
------------------------------

Hmm note that with dynamic running do not have gensteps ahead of time with which 
to create the pre-sized OpticksEvent.

**/

void CManager::BeginOfEventAction(const G4Event* event)
{
    LOG(LEVEL); 


    NPY<float>* gs = nullptr ;  // liable to take a while to get this to work 
    bool cfg4evt = true ; 
    m_ok->createEvent(gs, cfg4evt); 

    OpticksEvent* evt = m_ok->getG4Event();
    assert(evt); 
    initEvent(evt);    // configure event recording 

    m_ctx->setEvent(event);
}

void CManager::EndOfEventAction(const G4Event*)
{
    LOG(LEVEL); 
}

void CManager::PreUserTrackingAction(const G4Track* track)
{
    LOG(LEVEL); 
    m_ctx->setTrack(track);


    if(m_ctx->_optical)
    {   
        preTrack();
    }   
}

void CManager::PostUserTrackingAction(const G4Track* track)
{
    LOG(LEVEL); 

    int track_id = CTrack::Id(track) ;
    assert( track_id == m_ctx->_track_id );
    assert( track == m_ctx->_track );


    if(m_ctx->_optical)
    {   
        postTrack();
    }
}


void CManager::CManager::preTrack()
{
    if(m_engine)
    {
        m_engine->preTrack();
    }
}

void CManager::postTrack()
{

    if(m_ctx->_optical)
    {
        m_recorder->postTrack();
    } 
    if(m_engine)
    {
        m_engine->postTrack();
    }
}







void CManager::postpropagate()
{
    if(m_engine) m_engine->postpropagate();  
}


/**
CManager::addRandomNote
--------------------------

The note is associated with the index of the last random consumption, see boostrap/BLog.cc

**/

void CManager::addRandomNote(const char* note, int value)
{
    assert( m_engine ); 
    m_engine->addNote(note, value); 
}

void CManager::addRandomCut(const char* ckey, double cvalue)
{
    assert( m_engine ); 
    m_engine->addCut(ckey, cvalue); 
}







/**
CSteppingAction::UserSteppingAction
-------------------------------------

Invoked from tail of G4SteppingManager::Stepping (g4-cls G4SteppingManager), 
after InvokePostStepDoItProcs (g4-cls G4SteppingManager2).

Action depends on the boolean "done" result of CSteppingAction::setStep.
When done=true this stops tracking, which happens for absorption and truncation.

When not done the CProcessManager::ClearNumberOfInteractionLengthLeft is normally
invoked which results in the RNG for AB and SC being cleared.  
This forces G4VProcess::ResetNumberOfInteractionLengthLeft for every step, 
as that matches the way Opticks works `om-cls propagate`
with AB and SC RNG consumption at every "propagate_to_boundary".

The "--dbgskipclearzero" option inhibits this zeroing in the case of ZeroSteps
(which I think happen at boundary reflect or transmit) 

* hmm is OpBoundary skipped because its usually the winner process ? 
  so the standard G4VDiscreteProcess::PostStepDoIt will do the RNG consumption without assistance ?

See :doc:`stepping_process_review`

**/


void CManager::UserSteppingAction(const G4Step* step)
{
    LOG(LEVEL); 
    bool done = setStep(step);

    postStep();

    if(done)
    { 
        G4Track* track = step->GetTrack();    // m_track is const qualified
        track->SetTrackStatus(fStopAndKill);
    }
    else
    {
        prepareForNextStep(step);
    } 
}



void CManager::postStep()
{
    if(m_engine)
    {
        m_engine->postStep();
    }
}



/**
CManager::setStep
-------------------------

For a look into Geant4 ZeroStepping see notes/issues/review_alignment.rst 

**/

bool CManager::setStep(const G4Step* step)
{
    int noZeroSteps = -1 ;
    int severity = m_nav->SeverityOfZeroStepping( &noZeroSteps );

    if(noZeroSteps > 0)
    LOG(debug) 
              << " noZeroSteps " << noZeroSteps
              << " severity " << severity
              << " ctx " << m_ctx->desc()
              ;


    if(m_dbgflat)
    {
        addRandomNote("noZeroSteps", noZeroSteps ); 
    }


    bool done = false ; 

    m_ctx->setStep(step, noZeroSteps);
 
    if(m_ctx->_optical)
    {
        done = m_recorder->Record(m_ctx->_boundary_status);  
    }
    else
    {
        m_steprec->collectStep(step, m_ctx->_step_id);
    
        G4TrackStatus track_status = m_ctx->_track->GetTrackStatus(); 

        if(track_status == fStopAndKill)
        {
            done = true ;  
            m_steprec->storeStepsCollected(m_ctx->_event_id, m_ctx->_track_id, m_ctx->_pdg_encoding);
            m_steprec_store_count = m_steprec->getStoreCount(); 
        }
    }

   if(m_ctx->_step_total % 10000 == 0) 
       LOG(debug) << "CSA (totals%10k)"
                 << " track_total " <<  m_ctx->_track_total
                 << " step_total " <<  m_ctx->_step_total
                 ;

    return done ;  // (*lldb*) setStep
}

/**
CSteppingAction::prepareForNextStep
--------------------------------------

Clearing of the Number of Interation Lengths left is done in order to 
match Opticks random consumption done propagate_to_boundary 

See notes/issues/ts19-100.rst

**/

void CManager::prepareForNextStep(const G4Step* step)
{
    bool zeroStep = m_ctx->_noZeroSteps > 0 ;   // usually means there was a jump back 
    bool skipClear0 = zeroStep && m_ok->isDbgSkipClearZero()  ;  // --dbgskipclearzero
    bool skipClear = skipClear0 ; 
    int cursor = -1 ; 
 
    if( m_engine )
    {
        int currentStepFlatCount = m_engine->getCurrentStepFlatCount() ; 
        int currentRecordFlatCount = m_engine->getCurrentRecordFlatCount() ; 
        cursor = m_engine->getCursor() ; 
        int consumption_since_clear = m_cursor_at_clear > -1  ? cursor - m_cursor_at_clear : -1  ;  

        skipClear = consumption_since_clear == 3  ;  

        if(m_dbgflat) 
        {
            LOG(LEVEL) 
                << " cursor " <<  cursor
                << " m_cursor_at_clear " <<  m_cursor_at_clear
                << " consumption_since_clear " << consumption_since_clear 
                << " currentStepFlatCount " <<  currentStepFlatCount
                << " currentRecordFlatCount " <<  currentRecordFlatCount
                << " m_ctx._noZeroSteps " << m_ctx->_noZeroSteps 
                << " skipClear0 " << skipClear0
                <<  ( skipClear ? " SKIPPING CLEAR " : " proceed with CProcessManager::ClearNumberOfInteractionLengthLeft " )
                ; 

            if(m_ok->hasMask())   // --mask 
            {
                LOG(debug) 
                    << "[--mask] CProcessManager::ClearNumberOfInteractionLengthLeft " 
                    << " preStatus " << CStepStatus::Desc(step->GetPreStepPoint()->GetStepStatus())
                    << " postStatus " << CStepStatus::Desc(step->GetPostStepPoint()->GetStepStatus())
                    ; 
            }
        }
    }

    // if only 3 (OpBoundary,OpRayleigh,OpAbsorption) are primed with no consumption beyond 
    // propagate_to_boundary/DefinePhysicalStepLength 
    // THIS MAY BE A BETTER WAY OF CONTROLLING THE CLEAR  

    if(!skipClear)
    {  
        CProcessManager::ClearNumberOfInteractionLengthLeft( m_ctx->_process_manager, *m_ctx->_track, *m_ctx->_step );
        m_cursor_at_clear = cursor ; 
    }
}



void CManager::report(const char* msg)
{
    LOG(info) << msg ;
    std::cout 
           << " event_total " <<  m_ctx->_event_total << std::endl 
           << " track_total " <<  m_ctx->_track_total << std::endl 
           << " step_total " <<  m_ctx->_step_total << std::endl 
           ;
    //m_recorder->report(msg);
}





