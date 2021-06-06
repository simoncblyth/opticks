
#include "G4TransportationManager.hh"


#include "Randomize.hh"
#include "Opticks.hh"
#include "OpticksEvent.hh"

#include "CEvent.hh"
#include "CTrack.hh"
#include "CRandomEngine.hh"
#include "CRecorder.hh"
#include "CStep.hh"
#include "CProcessSubType.hh"

#include "CStepRec.hh"
#include "CStepStatus.hh"
#include "CProcessManager.hh"
#include "CCtx.hh"
#include "CManager.hh"

#include "PLOG.hh"

const plog::Severity CManager::LEVEL = PLOG::EnvLevel("CManager", "DEBUG"); 

CManager* CManager::fINSTANCE = nullptr ; 
CManager* CManager::Get(){  return fINSTANCE ; } 


CRecorder* CManager::getRecorder() const 
{ 
    return m_recorder ; 
}
CStepRec*  CManager::getStepRec() const 
{
    return m_noprec ; 
}



CRandomEngine* CManager::getRandomEngine() const 
{ 
    return m_engine ; 
}
CCtx& CManager::getCtx()
{
    return *m_ctx ; 
}

double CManager::flat_instrumented(const char* file, int line)
{
    return m_engine ? m_engine->flat_instrumented(file, line) : G4UniformRand() ; 
}

unsigned long long CManager::getSeqHis() const { return m_recorder->getSeqHis() ; }

/**
CManager::CManager
--------------------

--managermode

0
   return immediately from lifecycle calls doing nothing but logging
1 
   keep the CCtx updated to follow the propagation, but take no actions 

   * do not create OpticksEvent 
   * make no changes to the propagation 
   * do no recording



**/

CManager::CManager(Opticks* ok)
    :
    m_ok(ok),
    m_mode(ok->getManagerMode()),
    m_ctx(new CCtx(m_ok)),
    m_engine(m_ok->isAlign() ? new CRandomEngine(this) : NULL  ),   // --align
    m_recorder(new CRecorder(*m_ctx)),  // optical recording 
    m_noprec(new CStepRec(m_ok)),      // non-optical recording 
    m_dbgflat(m_ok->isDbgFlat()), 
    m_dbgrec(m_ok->isDbgRec()),
    m_trman(NULL),
    m_nav(NULL),
    m_noprec_store_count(0), 
    m_cursor_at_clear(-1)
{
    fINSTANCE = this ; 
}


void CManager::setMaterialBridge(const CMaterialBridge* material_bridge)
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    m_recorder->setMaterialBridge(material_bridge); 

    m_trman = G4TransportationManager::GetTransportationManager(); 
    m_nav = m_trman->GetNavigatorForTracking() ;
}


void CManager::BeginOfRunAction(const G4Run*)
{
    LOG(LEVEL) << " m_mode " << m_mode ;
}
void CManager::EndOfRunAction(const G4Run*)
{
    LOG(LEVEL) << " m_mode " << m_mode ;
}


/**
CManager::BeginOfEventAction
------------------------------

Hmm note that with dynamic running do not have gensteps ahead of time with which 
to create the pre-sized OpticksEvent.

**/

void CManager::BeginOfEventAction(const G4Event* event)
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    m_ctx->setEvent(event);

    if(m_ok->isSave()) presave();   // creates the OpticksEvent

    if( m_ctx->_number_of_input_photons  > 0 ) 
    {   
        LOG(LEVEL) 
            << " mocking BeginOfGenstep as have input photon primaries " 
            << CEvent::DescPrimary(event) 
            ;

        unsigned genstep_index = 0 ;  
        BeginOfGenstep(genstep_index, 'T', m_ctx->_number_of_input_photons, 0 );   
    }   
}


/**
CManager::EndOfEventAction
----------------------------

**/

void CManager::EndOfEventAction(const G4Event*)
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    LOG(LEVEL) 
        << " _number_of_input_photons " << m_ctx->_number_of_input_photons  
        ; 

    if(m_ok->isSave()) save() ; 

    

}


/**
CManager::BeginOfGenstep
-------------------------

Invoked by G4OpticksRecorder::BeginOfGenstep which is canonically placed 
just prior to the C/S generation loop

**/

void CManager::BeginOfGenstep(unsigned genstep_index, char gentype, int num_photons, int offset )
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    LOG(LEVEL) << " gentype " << gentype << " num_photons " << num_photons ; 

    m_ctx->BeginOfGenstep(genstep_index, gentype, num_photons, offset);  

    if(m_mode == 1 ) return ; 
    m_recorder->BeginOfGenstep();  
}



/**
CManager::presave
-------------------

Invoked from CManager::BeginOfEventAction, prepares for saving:

1. OpticksRun::m_g4evt OpticksEvent is created 
2. event recording is configured.

Hmm when using input_photons could use the carrier gensteps
to size the event and operate just like the OK event. 
But in ordinary Geant4 running only have genstep counts 
one by one. So need to find a workable way to extend the G4 event.

**/
void CManager::presave()
{
    LOG(LEVEL) << " mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    unsigned tagoffset = m_ctx->_event_id  ; 
    char ctrl = '-' ; 

    LOG(LEVEL) 
        << " [--save] creating OpticksEvent  " 
        << " m_ctx->_event_id(tagoffset) " << tagoffset 
        << " ctrl [" << ctrl << "]" 
        ; 


    if(m_mode == 1 ) return ; 
    m_ok->createEvent(tagoffset, ctrl);   

    OpticksEvent* evt = m_ok->getEvent(ctrl);
    assert(evt); 
    initEvent(evt);    // configure event recording 
}

/**
CManager::initEvent : configure event recording, limits/shapes etc.. 
------------------------------------------------------------------------

Invoked from CManager::BeginOfEventAction/CManager::presave

**/

void CManager::initEvent(OpticksEvent* evt)
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    assert( m_mode > 1 ); 

    m_ctx->initEvent(evt);
    m_recorder->initEvent(evt);

    NPY<float>* nopstep = evt->getNopstepData();
    if(!nopstep) LOG(fatal) << " nopstep NULL " << " evt " << evt->getShapeString() ; 
    assert(nopstep); 
    m_noprec->initEvent(nopstep);
}

/**
CManager::save
-----------------

Invoked from CManager::EndOfEventAction

**/

void CManager::save()
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    unsigned numPhotons = m_ctx->getNumPhotons() ;  // from CGenstepCollector
    LOG(LEVEL) << " m_mode " << m_mode << " numPhotons " << numPhotons ; 

    if(m_mode == 1 ) return ; 
    char ctrl = '-' ; 
    OpticksEvent* g4evt = m_ok->getEvent(ctrl) ; 

    if(g4evt)
    {
        LOG(LEVEL) << " --save g4evt numPhotons " << numPhotons ; 
        bool resize = false ; 
        g4evt->setNumPhotons( numPhotons, resize );  // hmm: probably not needed any more 

        m_ok->saveEvent(ctrl);
        m_ok->resetEvent(ctrl);
    }
}




void CManager::PreUserTrackingAction(const G4Track* track)
{
    //LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    m_ctx->setTrack(track);

    if(m_ctx->_optical)
    {   
        preTrack();
    }   
}

void CManager::PostUserTrackingAction(const G4Track* track)
{
    //LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    int track_id = CTrack::Id(track) ;
    assert( track_id == m_ctx->_track_id );

    if(m_ctx->_optical)
    {   
        postTrack();
    }
}


void CManager::CManager::preTrack()
{
    //LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

    if(m_engine)
    {
        m_engine->preTrack();
    }
}

void CManager::postTrack()
{
    //LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 


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
    //LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

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
    LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 


    bool done = setStep(step);

    postStep();


    G4Track* track = step->GetTrack();    // m_track is const qualified

    if(done)
    { 
        track->SetTrackStatus(fStopAndKill);
    }
    else
    {
        prepareForNextStep(step, track);
    } 
}



void CManager::postStep()
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    if(m_mode == 0 ) return ; 

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
    LOG(LEVEL) << " m_mode " << m_mode ;
    assert(m_mode > 0 ); 

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


    m_ctx->setStep(step, noZeroSteps);


    bool done = false ; 

    if(m_mode == 1)
    {
         done = m_ctx->_track_status == fStopAndKill ; 
    }
    else
    {
        if(m_ctx->_optical)
        {
            if(m_ctx->_boundary_status == 0 ) LOG(fatal) << " boundary_status zero " ; 
            done = m_recorder->Record(m_ctx->_boundary_status);  
        }
        else
        {
            m_noprec->collectStep(step, m_ctx->_step_id);

            if(m_ctx->_track_status == fStopAndKill)
            {
                done = true ;  
                m_noprec->storeStepsCollected(m_ctx->_event_id, m_ctx->_track_id, m_ctx->_pdg_encoding);
                m_noprec_store_count = m_noprec->getStoreCount(); 
            }
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

void CManager::prepareForNextStep(const G4Step* step, G4Track* mtrack)
{
    LOG(LEVEL) << " m_mode " << m_mode ;
    assert( m_mode > 0 );  


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
        CProcessManager::ClearNumberOfInteractionLengthLeft( m_ctx->_process_manager, *mtrack, *m_ctx->_step );
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



