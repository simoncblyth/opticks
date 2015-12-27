#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;
class OpSource ; 

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction(OpSource* generator);
    virtual ~PrimaryGeneratorAction();

  public:
    virtual void GeneratePrimaries(G4Event*);

  private:
    //G4VPrimaryGenerator*  m_generator ;
    OpSource*  m_generator ;

};

inline PrimaryGeneratorAction::PrimaryGeneratorAction(OpSource* generator)
    : 
    G4VUserPrimaryGeneratorAction(), 
    m_generator(generator)
{
}
