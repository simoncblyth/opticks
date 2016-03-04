#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;
class CSource ; 

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction(CSource* generator);
    virtual ~PrimaryGeneratorAction();

  public:
    virtual void GeneratePrimaries(G4Event*);

  private:
    //G4VPrimaryGenerator*  m_generator ;
    CSource*  m_generator ;

};

inline PrimaryGeneratorAction::PrimaryGeneratorAction(CSource* generator)
    : 
    G4VUserPrimaryGeneratorAction(), 
    m_generator(generator)
{
}
