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

#include "PrimaryGeneratorAction.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleGun.hh"

#ifdef WITH_OPTICKS
#include "G4Opticks.hh"
#endif





using CLHEP::MeV ; 

PrimaryGeneratorAction::PrimaryGeneratorAction(Ctx* ctx_)
    :
    G4VUserPrimaryGeneratorAction(),
    ctx(ctx_),
    fParticleGun(new G4ParticleGun(1))
{
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particle = particleTable->FindParticle("e+");
    fParticleGun->SetParticleDefinition(particle);
    fParticleGun->SetParticleTime(0.0*CLHEP::ns);
    fParticleGun->SetParticlePosition(G4ThreeVector(0.0*CLHEP::cm,0.0*CLHEP::cm,0.0*CLHEP::cm));
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(1.,0.,0.));
    fParticleGun->SetParticleEnergy(0.8*MeV);   // few photons at ~0.7*MeV loads from ~ 0.8*MeV
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    fParticleGun->GeneratePrimaryVertex(anEvent);

#ifdef WITH_OPTICKS
    // not strictly required by Opticks, but useful for debugging + visualization of non-opticals
    //G4Opticks::GetOpticks()->collectPrimaries(anEvent) ;  
#endif

}


 
