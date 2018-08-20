// g4-

#include "CFG4_PUSH.hh"




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
#include "CStepRec.hh"
#include "CStp.hh"
#include "CSteppingAction.hh"
#include "CProcessManager.hh"
#include "CStepStatus.hh"
#include "CTrack.hh"
#include "CG4Ctx.hh"
#include "CG4.hh"

#include "CFG4_POP.hh"


// optickscore-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

// npy-
#include "PLOG.hh"


/**
CSteppingAction
=================

Canonical instance (m_sa) is ctor resident of CG4 

**/

CSteppingAction::CSteppingAction(CG4* g4, bool dynamic)
   : 
   G4UserSteppingAction(),
   m_g4(g4),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),
   m_dbgrec(m_ok->isDbgRec()),
   m_dynamic(dynamic),
   m_geometry(g4->getGeometry()),
   m_material_bridge(NULL),
   m_mlib(g4->getMaterialLib()),
   m_recorder(g4->getRecorder()),
   m_steprec(g4->getStepRec()),
   m_steprec_store_count(0)
{ 
}

void CSteppingAction::postinitialize()
{
   // called from CG4::postinitialize
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);


    m_trman = G4TransportationManager::GetTransportationManager(); 
    m_nav = m_trman->GetNavigatorForTracking() ;

}

CSteppingAction::~CSteppingAction()
{ 
}

void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    bool done = setStep(step);

    m_g4->postStep();


    if(done)
    { 
        G4Track* track = step->GetTrack();    // m_track is const qualified
        track->SetTrackStatus(fStopAndKill);
        // stops tracking when reach truncation as well as absorption
    }
    else
    {
        // guess work for alignment
        // should this be done after a jump ?

        bool zeroStep = m_ctx._noZeroSteps > 0 ;   // usually means there was a jump back 
        bool skipClear = zeroStep && m_ok->isDbgSkipClearZero()  ;

        if(skipClear)
        {
            LOG(error) << " --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft " ; 
        }  
        else
        {
            CProcessManager::ClearNumberOfInteractionLengthLeft( m_ctx._process_manager, *m_ctx._track, *m_ctx._step );
        }

        if(m_ok->hasMask())
        {
            LOG(debug) << "[--mask] CSteppingAction::UserSteppingAction CProcessManager::ClearNumberOfInteractionLengthLeft " 
                      << " preStatus " << CStepStatus::Desc(step->GetPreStepPoint()->GetStepStatus())
                      << " postStatus " << CStepStatus::Desc(step->GetPostStepPoint()->GetStepStatus())
                      ; 
        }
    } 
}



/*
At the end of everystep the RNG for AB and SC are cleared, in order to 
force G4VProcess::ResetNumberOfInteractionLengthLeft for every step, as
that is how Opticks works with AB and SC RNG consumption at every "propagate_to_boundary".

* hmm is OpBoundary skipped because its usually the winner process ? 
  so the standard G4VDiscreteProcess::PostStepDoIt will do the RNG consumption without assistance ?

See :doc:`stepping_process_review`
*/





bool CSteppingAction::setStep(const G4Step* step)
{
    int noZeroSteps = -1 ;
    int severity = m_nav->SeverityOfZeroStepping( &noZeroSteps );
    if(noZeroSteps > 0)
    LOG(info) 
              << " noZeroSteps " << noZeroSteps
              << " severity " << severity
              << " ctx " << m_ctx.desc()
              ;


    bool done = false ; 

    m_ctx.setStep(step, noZeroSteps);
 
    if(m_ctx._optical)
    {
        done = m_recorder->Record(m_ctx._boundary_status);  
    }
    else
    {
        m_steprec->collectStep(step, m_ctx._step_id);
    
        G4TrackStatus track_status = m_ctx._track->GetTrackStatus(); 

        if(track_status == fStopAndKill)
        {
            done = true ;  
            m_steprec->storeStepsCollected(m_ctx._event_id, m_ctx._track_id, m_ctx._pdg_encoding);
            m_steprec_store_count = m_steprec->getStoreCount(); 
        }
    }

   if(m_ctx._step_total % 10000 == 0) 
       LOG(debug) << "CSA (totals%10k)"
                 << " track_total " <<  m_ctx._track_total
                 << " step_total " <<  m_ctx._step_total
                 ;

    return done ;  // (*lldb*) setStep
}



void CSteppingAction::report(const char* msg)
{
    LOG(info) << msg ;
    std::cout 
           << " event_total " <<  m_ctx._event_total << std::endl 
           << " track_total " <<  m_ctx._track_total << std::endl 
           << " step_total " <<  m_ctx._step_total << std::endl 
           ;
    //m_recorder->report(msg);
}

