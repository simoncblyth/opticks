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


#include <sstream>
#include <iomanip>

#include "G4EventManager.hh"
#include "G4TrackingManager.hh"
#include "G4SteppingManager.hh"

#include "CProcess.hh"


std::string CProcess::Desc( G4VProcess* proc )
{
    if(!proc) return "" ; 

    std::stringstream ss ; 

    ss << "CPro "
       << std::setw(15) << proc->GetProcessName()
       << " LenLeft " << std::setw(10) << proc->GetNumberOfInteractionLengthLeft() 
       << " LenTrav " << std::setw(10) << proc->GetTotalNumberOfInteractionLengthTraversed() 
       << " AtRest/AlongStep/PostStep " 
       << ( proc->isAtRestDoItIsEnabled()    ? "Y" : "N" )
       << ( proc->isAlongStepDoItIsEnabled() ? "Y" : "N" )
       << ( proc->isPostStepDoItIsEnabled()  ? "Y" : "N" )

/*
       << " NumberOfInteractionLengthLeft " << std::setw(10) << proc->GetNumberOfInteractionLengthLeft() 
       << " isAtRestDoItIsEnabled " << ( proc->isAtRestDoItIsEnabled() ? "Y" : "N" )
       << " isAlongStepDoItIsEnabled " << ( proc->isAlongStepDoItIsEnabled() ? "Y" : "N" )
       << " isPostStepDoItIsEnabled " << ( proc->isPostStepDoItIsEnabled() ? "Y" : "N" )
*/
       ;

    return ss.str();
} 



G4VProcess* CProcess::CurrentProcess()
{
    G4EventManager* evtMgr = G4EventManager::GetEventManager() ;
    G4TrackingManager* trkMgr = evtMgr->GetTrackingManager() ; 
    G4SteppingManager* stepMgr = trkMgr->GetSteppingManager() ; 
    return stepMgr->GetfCurrentProcess() ; 
}





