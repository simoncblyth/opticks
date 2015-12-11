
#pragma once

//  /usr/local/env/g4/geant4.10.02.install/share/Geant4-10.2.0/examples/extended/optical/OpNovice/include

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4ParticleGun;
class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();
    virtual ~PrimaryGeneratorAction();

  public:
    virtual void GeneratePrimaries(G4Event*);

    void SetOpticalPhotonPolarization();
    void SetOpticalPhotonPolarization(G4double);

  private:
    G4ParticleGun*  m_gun ;
};


