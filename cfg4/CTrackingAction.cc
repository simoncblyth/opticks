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

// g4-

#include "CFG4_PUSH.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"
#include "CFG4_POP.hh"

// cg4-
#include "CManager.hh"
#include "CTrackingAction.hh"


#include "PLOG.hh"

const plog::Severity CTrackingAction::LEVEL = PLOG::EnvLevel("CTrackingAction", "DEBUG") ; 


/**
CTrackingAction
=================

Canonical instance (m_ta) is ctor resident of CG4 

**/

CTrackingAction::CTrackingAction(CManager* manager)
   : 
   G4UserTrackingAction(),
   m_manager(manager)
{ 
}

CTrackingAction::~CTrackingAction()
{ 
}

/**
CTrackingAction::PreUserTrackingAction
-----------------------------------------

Invoked by G4TrackingManager::ProcessOneTrack immediately after:: 

   G4SteppingManager::SetInitialStep(G4Track* valueTrack)
   G4Step::InitializeStep( G4Track* aValue )
   
   g4-;g4-cls G4TrackingManager
   g4-;g4-cls G4SteppingManager
   g4-;g4-cls G4Step

**/

void CTrackingAction::PreUserTrackingAction(const G4Track* track)
{
    m_manager->PreUserTrackingAction(track); 
}

void CTrackingAction::PostUserTrackingAction(const G4Track* track)
{
    m_manager->PostUserTrackingAction(track); 
}


