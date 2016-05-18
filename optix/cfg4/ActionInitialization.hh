#pragma once

class G4VUserPrimaryGeneratorAction ;
class G4UserSteppingAction ;

#include "G4VUserActionInitialization.hh"

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(G4VUserPrimaryGeneratorAction* pga, G4UserSteppingAction* sa);
    virtual ~ActionInitialization();

    virtual void Build() const;
    virtual G4VSteppingVerbose* InitializeSteppingVerbose() const; 

  private:
    G4VUserPrimaryGeneratorAction* m_pga ;  
    G4UserSteppingAction*          m_sa ; 

};


inline ActionInitialization::ActionInitialization(G4VUserPrimaryGeneratorAction* pga, G4UserSteppingAction* sa)
    : 
    G4VUserActionInitialization(), 
    m_pga(pga),
    m_sa(sa)
{}



