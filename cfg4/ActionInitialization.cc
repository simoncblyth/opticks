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

#include "CFG4_BODY.hh"
#include "ActionInitialization.hh"

//#include "PrimaryGeneratorAction.hh"
//#include "G4UserSteppingAction.hh"

#include "CFG4_PUSH.hh"
#include "SteppingVerbose.hh"
#include "CFG4_POP.hh"


ActionInitialization::ActionInitialization(
       G4VUserPrimaryGeneratorAction* pga, 
       G4UserSteppingAction* sa,
       G4UserTrackingAction* ta,
       G4UserRunAction* ra,
       G4UserEventAction* ea
)
    : 
    G4VUserActionInitialization(), 
    m_pga(pga),
    m_sa(sa),
    m_ta(ta),
    m_ra(ra),
    m_ea(ea)
{}


ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
    SetUserAction(m_pga);
    SetUserAction(m_sa);
    SetUserAction(m_ta);
    SetUserAction(m_ra);
    SetUserAction(m_ea);
}

G4VSteppingVerbose* ActionInitialization::InitializeSteppingVerbose() const
{
  return new SteppingVerbose();
}  




