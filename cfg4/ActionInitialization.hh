#pragma once

class G4VUserPrimaryGeneratorAction ;
class G4UserSteppingAction ;
class G4UserRunAction ;
class G4UserEventAction ;

#include "G4VUserActionInitialization.hh"
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(
           G4VUserPrimaryGeneratorAction* pga, 
           G4UserSteppingAction* sa,
           G4UserRunAction* ra,
           G4UserEventAction* ea
    );

    virtual ~ActionInitialization();

    virtual void Build() const;
    virtual G4VSteppingVerbose* InitializeSteppingVerbose() const; 

  private:
    G4VUserPrimaryGeneratorAction* m_pga ;  
    G4UserSteppingAction*          m_sa ; 
    G4UserRunAction*               m_ra ; 
    G4UserEventAction*             m_ea ; 

};
#include "CFG4_TAIL.hh"


