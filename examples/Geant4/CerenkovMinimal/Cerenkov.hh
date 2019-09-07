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

#include "G4Cerenkov.hh"

struct Cerenkov : public G4Cerenkov 
{
    Cerenkov( const G4String& processName, G4ProcessType type = fElectromagnetic);

    G4bool IsApplicable(const G4ParticleDefinition& aParticleType);

    void BuildPhysicsTable(const G4ParticleDefinition& aParticleType);

    G4double PostStepGetPhysicalInteractionLength(
                                           const G4Track& aTrack,
                                           G4double ignored,
                                           G4ForceCondition* condition);

    G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep);

};


