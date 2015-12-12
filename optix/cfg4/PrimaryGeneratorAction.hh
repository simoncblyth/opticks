#pragma once

//  /usr/local/env/g4/geant4.10.02.install/share/Geant4-10.2.0/examples/extended/optical/OpNovice/include
//  /usr/local/env/g4/geant4.10.02.install/share/Geant4-10.2.0/examples/extended/optical/wls/src/WLSPrimaryGeneratorAction.cc

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();
    virtual ~PrimaryGeneratorAction();
  public:
    virtual void GeneratePrimaries(G4Event*);

    G4VPrimaryGenerator* MakeGenerator(G4int n=1);

  private:
    G4VPrimaryGenerator*  m_generator ;

};


