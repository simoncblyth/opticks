#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;
class TorchStepNPY ; 

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction(TorchStepNPY* torch);
    virtual ~PrimaryGeneratorAction();
  private:
    void init();

  public:
    virtual void GeneratePrimaries(G4Event*);

  private:
    G4VPrimaryGenerator*  m_generator ;
    TorchStepNPY*         m_torch ; 

};

inline PrimaryGeneratorAction::PrimaryGeneratorAction(TorchStepNPY* torch)
    : 
    G4VUserPrimaryGeneratorAction(), 
    m_generator(NULL),
    m_torch(torch)
{
    init();
}
