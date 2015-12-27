#pragma once

class PrimaryGeneratorAction ;
class SteppingAction ; 

#include "G4VUserActionInitialization.hh"

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(PrimaryGeneratorAction* pga, SteppingAction* sa);
    virtual ~ActionInitialization();

    virtual void Build() const;
    virtual G4VSteppingVerbose* InitializeSteppingVerbose() const; 

  private:
    PrimaryGeneratorAction* m_pga ;  
    SteppingAction*         m_sa ; 

};


inline ActionInitialization::ActionInitialization(PrimaryGeneratorAction* pga, SteppingAction* sa)
    : 
    G4VUserActionInitialization(), 
    m_pga(pga),
    m_sa(sa)
{}



