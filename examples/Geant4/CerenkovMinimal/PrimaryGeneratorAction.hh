#pragma once
#include "PrimaryGeneratorAction.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleGun.hh"
   
#include "G4VUserPrimaryGeneratorAction.hh"

struct Ctx ; 
class G4ParticleGun ; 

struct PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
    PrimaryGeneratorAction(Ctx* ctx_);
    void GeneratePrimaries(G4Event* anEvent);
    void collectPrimary(const G4Event* anEvent);
   
    Ctx*           ctx ; 
    G4ParticleGun* fParticleGun ; 
};


