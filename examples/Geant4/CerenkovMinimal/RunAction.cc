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


#include <cassert>

#ifdef WITH_OPTICKS
#include "G4TransportationManager.hh"
#include "G4Opticks.hh"
#endif

#include "RunAction.hh"

RunAction::RunAction(Ctx* ctx_) 
    :   
    G4UserRunAction(),
    ctx(ctx_)
{
}
void RunAction::BeginOfRunAction(const G4Run*)
{
#ifdef WITH_OPTICKS
    G4cout << "\n\n###[ RunAction::BeginOfRunAction G4Opticks.setGeometry\n\n" << G4endl ; 
    G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ; 
    assert( world ) ; 
    bool standardize_geant4_materials = true ;   // required for alignment 
    G4Opticks::GetOpticks()->setGeometry(world, standardize_geant4_materials );    
    G4cout << "\n\n###] RunAction::BeginOfRunAction G4Opticks.setGeometry\n\n" << G4endl ; 
#endif
}
void RunAction::EndOfRunAction(const G4Run*)
{
#ifdef WITH_OPTICKS
    G4cout << "\n\n###[ RunAction::EndOfRunAction G4Opticks.Finalize\n\n" << G4endl ; 
    G4Opticks::Finalize();
    G4cout << "\n\n###] RunAction::EndOfRunAction G4Opticks.Finalize\n\n" << G4endl ; 
#endif
}

