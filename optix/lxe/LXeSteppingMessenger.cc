#include "LXeSteppingMessenger.hh"
#include "LXeSteppingAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"


LXeSteppingMessenger::LXeSteppingMessenger(LXeSteppingAction* step)
 : fStepping(step)
{
  fOneStepPrimariesCmd = new G4UIcmdWithABool("/LXe/oneStepPrimaries",this);
  fOneStepPrimariesCmd->
      SetGuidance("Only allows primaries to go one step before being killed.");

}


LXeSteppingMessenger::~LXeSteppingMessenger(){
  delete fOneStepPrimariesCmd;
}


void 
LXeSteppingMessenger::SetNewValue(G4UIcommand* command,G4String newValue){
  if( command == fOneStepPrimariesCmd ){
    fStepping->SetOneStepPrimaries(fOneStepPrimariesCmd
                                  ->GetNewBoolValue(newValue));
  }
}
