#pragma once

#include "globals.hh"
#include "G4UImessenger.hh"

class OpNovicePhysicsList;
class G4UIdirectory;
class G4UIcmdWithAnInteger;

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
class CFG4_API OpNovicePhysicsListMessenger : public G4UImessenger
{
  public:
    OpNovicePhysicsListMessenger(OpNovicePhysicsList* );
    virtual ~OpNovicePhysicsListMessenger();
 
    virtual void SetNewValue(G4UIcommand*, G4String);
 
  private:
    OpNovicePhysicsList*  fPhysicsList;
 
    G4UIdirectory*        fOpNoviceDir;
    G4UIdirectory*        fPhysDir;
    G4UIcmdWithAnInteger* fVerboseCmd;
    G4UIcmdWithAnInteger* fCerenkovCmd;
};

#include "CFG4_TAIL.hh"

