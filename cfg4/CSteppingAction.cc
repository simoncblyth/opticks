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
}

CSteppingAction::~CSteppingAction()
{ 
}

void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    bool done = setStep(step);

    if(done)
    { 
        G4Track* track = step->GetTrack();    // m_track is const qualified
        track->SetTrackStatus(fStopAndKill);
        // stops tracking when reach truncation as well as absorption
    }
}

bool CSteppingAction::setStep(const G4Step* step)
{
    bool done = false ; 

    m_ctx.setStep(step);
 
    if(m_ctx._optical)
    {
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus boundary_status = CBoundaryProcess::GetOpBoundaryProcessStatus() ;
#else
        G4OpBoundaryProcessStatus boundary_status = CBoundaryProcess::GetOpBoundaryProcessStatus() ;
#endif
        done = m_recorder->Record(boundary_status);
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

    return done ;
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

