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


const plog::Severity CSteppingAction::LEVEL = PLOG::EnvLevel("CSteppingAction", "DEBUG") ; 


CSteppingAction::CSteppingAction(CG4* g4, bool dynamic)
   : 
   G4UserSteppingAction(),
   m_g4(g4),
   m_engine(m_g4->getRandomEngine()),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),
   m_dbgflat(m_ok->isDbgFlat()), 
   m_dbgrec(m_ok->isDbgRec()),
   m_dynamic(dynamic),
   m_geometry(g4->getGeometry()),
   m_material_bridge(NULL),
   m_mlib(g4->getMaterialLib()),
   m_recorder(g4->getRecorder()),
   m_steprec(g4->getStepRec()),
   m_trman(NULL),
   m_nav(NULL),
   m_steprec_store_count(0), 
   m_cursor_at_clear(-1)
{ 
}

/**
CSteppingAction::postinitialize
---------------------------------

Called from CG4::postinitialize

m_nav
    G4Navigator 


**/

void CSteppingAction::postinitialize()
{
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);


    m_trman = G4TransportationManager::GetTransportationManager(); 
    m_nav = m_trman->GetNavigatorForTracking() ;

}

CSteppingAction::~CSteppingAction()
{ 
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

void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    if(m_dbgflat) LOG(LEVEL) << "[" ; 

    bool done = setStep(step);

    m_g4->postStep();

    if(done)
    { 
        G4Track* track = step->GetTrack();    // m_track is const qualified
        track->SetTrackStatus(fStopAndKill);
    }
    else
    {
        prepareForNextStep(step);

    } 

    if(m_dbgflat) LOG(LEVEL) << "]" ; 
}


/**
CSteppingAction::prepareForNextStep
--------------------------------------

Clearing of the Number of Interation Lengths left is done in order to 
match Opticks random consumption done propagate_to_boundary 

See notes/issues/ts19-100.rst

**/

void CSteppingAction::prepareForNextStep(const G4Step* step)
{
    bool zeroStep = m_ctx._noZeroSteps > 0 ;   // usually means there was a jump back 
    bool skipClear0 = zeroStep && m_ok->isDbgSkipClearZero()  ;  // --dbgskipclearzero

    int currentStepFlatCount = m_engine->getCurrentStepFlatCount() ; 
    int currentRecordFlatCount = m_engine->getCurrentRecordFlatCount() ; 
    int cursor = m_engine->getCursor() ; 
    int consumption_since_clear = m_cursor_at_clear > -1  ? cursor - m_cursor_at_clear : -1  ;  

    bool skipClear1 = consumption_since_clear == 3  ;  

    //bool skipClear = skipClear0 ;  // old way 
    bool skipClear = skipClear1 ;  // new way 


    // if only 3 (OpBoundary,OpRayleigh,OpAbsorption) are primed with no consumption beyond 
    // propagate_to_boundary/DefinePhysicalStepLength 
    // THIS MAY BE A BETTER WAY OF CONTROLLING THE CLEAR  

    if(m_dbgflat) 
    {
        LOG(LEVEL) 
            << " cursor " <<  cursor
            << " m_cursor_at_clear " <<  m_cursor_at_clear
            << " consumption_since_clear " << consumption_since_clear 
            << " currentStepFlatCount " <<  currentStepFlatCount
            << " currentRecordFlatCount " <<  currentRecordFlatCount
            << " m_ctx._noZeroSteps " << m_ctx._noZeroSteps 
            << " skipClear0 " << skipClear0
            << " skipClear1 " << skipClear1
            <<  ( skipClear ? " SKIPPING CLEAR " : " proceed with CProcessManager::ClearNumberOfInteractionLengthLeft " )
            ; 
    }

    if(!skipClear)
    {  
        CProcessManager::ClearNumberOfInteractionLengthLeft( m_ctx._process_manager, *m_ctx._track, *m_ctx._step );
        m_cursor_at_clear = cursor ; 
    }


    if(m_ok->hasMask())   // --mask 
    {
        LOG(debug) 
            << "[--mask] CProcessManager::ClearNumberOfInteractionLengthLeft " 
            << " preStatus " << CStepStatus::Desc(step->GetPreStepPoint()->GetStepStatus())
            << " postStatus " << CStepStatus::Desc(step->GetPostStepPoint()->GetStepStatus())
            ; 
    }
}




/**
CSteppingAction::setStep
-------------------------

For a look into Geant4 ZeroStepping see notes/issues/review_alignment.rst 

**/

bool CSteppingAction::setStep(const G4Step* step)
{
    int noZeroSteps = -1 ;
    int severity = m_nav->SeverityOfZeroStepping( &noZeroSteps );

    if(noZeroSteps > 0)
    LOG(debug) 
              << " noZeroSteps " << noZeroSteps
              << " severity " << severity
              << " ctx " << m_ctx.desc()
              ;


    if(m_dbgflat)
    {
        m_g4->addRandomNote("noZeroSteps", noZeroSteps ); 
    }


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

