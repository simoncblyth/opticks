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


#include "G4EventManager.hh"
#include "G4TrackingManager.hh"
#include "G4SteppingManager.hh"

#include "CStepping.hh"


CSteppingState CStepping::CurrentState()
{
    G4EventManager* evtMgr = G4EventManager::GetEventManager() ;
    G4TrackingManager* trkMgr = evtMgr->GetTrackingManager() ; 
    G4SteppingManager* stepMgr = trkMgr->GetSteppingManager() ; 
 
    CSteppingState ss ; 
    ss.fPostStepGetPhysIntVector = stepMgr->GetfPostStepGetPhysIntVector();
    //ss.fSelectedPostStepDoItVector = stepMgr->GetfSelectedPostStepDoItVector() ;
    ss.fCurrentProcess = stepMgr->GetfCurrentProcess() ; 
    ss.MAXofPostStepLoops = stepMgr->GetMAXofPostStepLoops() ;
    ss.fStepStatus        = stepMgr->GetfStepStatus() ; 
    
    return ss ; 
}



