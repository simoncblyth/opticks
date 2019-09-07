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

class G4VUserPrimaryGeneratorAction ;
class G4UserSteppingAction ;
class G4UserTrackingAction ;
class G4UserRunAction ;
class G4UserEventAction ;

#include "G4VUserActionInitialization.hh"
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(
           G4VUserPrimaryGeneratorAction* pga, 
           G4UserSteppingAction* sa,
           G4UserTrackingAction* ta,
           G4UserRunAction* ra,
           G4UserEventAction* ea
    );

    virtual ~ActionInitialization();

    virtual void Build() const;
    virtual G4VSteppingVerbose* InitializeSteppingVerbose() const; 

  private:
    G4VUserPrimaryGeneratorAction* m_pga ;  
    G4UserSteppingAction*          m_sa ; 
    G4UserTrackingAction*          m_ta ; 
    G4UserRunAction*               m_ra ; 
    G4UserEventAction*             m_ea ; 

};
#include "CFG4_TAIL.hh"


