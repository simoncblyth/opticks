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

#include "globals.hh"
#include "G4UImessenger.hh"

class OpNovicePhysicsList;
class G4UIdirectory;
class G4UIcmdWithAnInteger;

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
class CFG4_API OpNovicePhysicsListMessenger : public G4UImessenger
{
  public:
    OpNovicePhysicsListMessenger(OpNovicePhysicsList* );
    virtual ~OpNovicePhysicsListMessenger();
 
    virtual void SetNewValue(G4UIcommand*, G4String);
 
  private:
    OpNovicePhysicsList*  fPhysicsList;
 
    G4UIdirectory*        fOpNoviceDir;
    G4UIdirectory*        fPhysDir;
    G4UIcmdWithAnInteger* fVerboseCmd;
    G4UIcmdWithAnInteger* fCerenkovCmd;
};

#include "CFG4_TAIL.hh"

