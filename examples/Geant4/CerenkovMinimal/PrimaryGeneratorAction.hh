#pragma once

// generator
#include "G4VUserPrimaryGeneratorAction.hh"

class G4ParticleGun ; 

struct PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
    PrimaryGeneratorAction();
    void GeneratePrimaries(G4Event* anEvent);
   
    G4ParticleGun* fParticleGun ; 
};


   

