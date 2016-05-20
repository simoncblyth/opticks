#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;
class CSource ; 

class CPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    CPrimaryGeneratorAction(CSource* generator);
    virtual ~CPrimaryGeneratorAction();

  public:
    virtual void GeneratePrimaries(G4Event*);

  private:
    //G4VPrimaryGenerator*  m_generator ;
    CSource*  m_generator ;

};

inline CPrimaryGeneratorAction::CPrimaryGeneratorAction(CSource* generator)
    : 
    G4VUserPrimaryGeneratorAction(), 
    m_generator(generator)
{
}
