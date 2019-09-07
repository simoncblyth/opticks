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

#pragma once



#include <string>
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CProcessManager
=================

::

   g4-;g4-cls G4ProcessManager

**/

class G4ProcessManager ;
class G4VDiscreteProcess ;
class G4Track ;
class G4Step ;


struct CFG4_API CProcessManager
{
    static std::string Desc(G4ProcessManager* proc) ;  
    static G4ProcessManager* Current(G4Track* trk) ;  
    static void ClearNumberOfInteractionLengthLeft(G4ProcessManager* proMgr, const G4Track& aTrack, const G4Step& aStep);
    static void ResetNumberOfInteractionLengthLeft(G4ProcessManager* proMgr);

    static G4VDiscreteProcess* GetDiscreteProcess( G4ProcessManager* proMgr, const char* name);
    static void ClearNumberOfInteractionLengthLeft(G4ProcessManager* proMgr, const G4Track& aTrack, const G4Step& aStep, const char* name);



};

#include "CFG4_TAIL.hh"
 
