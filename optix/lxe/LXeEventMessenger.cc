#include "LXeEventMessenger.hh"
#include "LXeEventAction.hh"

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"


LXeEventMessenger::LXeEventMessenger(LXeEventAction* event)
 : fLXeEvent(event)
{
  fSaveThresholdCmd = new G4UIcmdWithAnInteger("/LXe/saveThreshold",this);
  fSaveThresholdCmd->
SetGuidance("Set the photon count threshold for saving the random number seed");
  fSaveThresholdCmd->SetParameterName("photons",true);
  fSaveThresholdCmd->SetDefaultValue(4500);
  fSaveThresholdCmd->AvailableForStates(G4State_PreInit,G4State_Idle);

  fVerboseCmd = new G4UIcmdWithAnInteger("/LXe/eventVerbose",this);
  fVerboseCmd->SetGuidance("Set the verbosity of event data.");
  fVerboseCmd->SetParameterName("verbose",true);
  fVerboseCmd->SetDefaultValue(1);

  fPmtThresholdCmd = new G4UIcmdWithAnInteger("/LXe/pmtThreshold",this);
  fPmtThresholdCmd->SetGuidance("Set the pmtThreshold (in # of photons)");

  fForceDrawPhotonsCmd=new G4UIcmdWithABool("/LXe/forceDrawPhotons",this);
  fForceDrawPhotonsCmd->SetGuidance("Force drawing of photons.");
  fForceDrawPhotonsCmd
    ->SetGuidance("(Higher priority than /LXe/forceDrawNoPhotons)");

  fForceDrawNoPhotonsCmd=new G4UIcmdWithABool("/LXe/forceDrawNoPhotons",this);
  fForceDrawNoPhotonsCmd->SetGuidance("Force no drawing of photons.");
  fForceDrawNoPhotonsCmd
    ->SetGuidance("(Lower priority than /LXe/forceDrawPhotons)");
}


LXeEventMessenger::~LXeEventMessenger(){
  delete fSaveThresholdCmd;
  delete fVerboseCmd;
  delete fPmtThresholdCmd;
  delete fForceDrawPhotonsCmd;
  delete fForceDrawNoPhotonsCmd;
}


void LXeEventMessenger::SetNewValue(G4UIcommand* command, G4String newValue){
  if( command == fSaveThresholdCmd ){
    fLXeEvent->SetSaveThreshold(fSaveThresholdCmd->GetNewIntValue(newValue));
  }
  else if( command == fVerboseCmd ){
    fLXeEvent->SetEventVerbose(fVerboseCmd->GetNewIntValue(newValue));
  }
  else if( command == fPmtThresholdCmd ){
    fLXeEvent->SetPMTThreshold(fPmtThresholdCmd->GetNewIntValue(newValue));
  }
  else if(command == fForceDrawPhotonsCmd){
    fLXeEvent->SetForceDrawPhotons(fForceDrawPhotonsCmd
                                  ->GetNewBoolValue(newValue));
  }
  else if(command == fForceDrawNoPhotonsCmd){
    fLXeEvent->SetForceDrawNoPhotons(fForceDrawNoPhotonsCmd
                                  ->GetNewBoolValue(newValue));
    G4cout<<"TEST"<<G4endl;
  }
}
