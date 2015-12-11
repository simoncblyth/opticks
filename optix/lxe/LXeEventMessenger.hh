#ifndef LXeEventMessenger_h
#define LXeEventMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class LXeEventAction;
class G4UIcmdWithAnInteger;
class G4UIcmdWithABool;

class LXeEventMessenger: public G4UImessenger
{
  public:

    LXeEventMessenger(LXeEventAction*);
    virtual ~LXeEventMessenger();
 
    virtual void SetNewValue(G4UIcommand*, G4String);
 
  private:

    LXeEventAction*        fLXeEvent;
    G4UIcmdWithAnInteger*  fSaveThresholdCmd;
    G4UIcmdWithAnInteger*  fVerboseCmd;
    G4UIcmdWithAnInteger*  fPmtThresholdCmd;
    G4UIcmdWithABool*      fForceDrawPhotonsCmd;
    G4UIcmdWithABool*      fForceDrawNoPhotonsCmd;
};

#endif
