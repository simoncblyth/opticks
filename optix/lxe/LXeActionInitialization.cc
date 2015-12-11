#include "LXeActionInitialization.hh"

#include "LXePrimaryGeneratorAction.hh"

#include "LXeRunAction.hh"
#include "LXeEventAction.hh"
#include "LXeTrackingAction.hh"
#include "LXeSteppingAction.hh"
#include "LXeStackingAction.hh"
#include "LXeSteppingVerbose.hh"

#include "LXeRecorderBase.hh"


LXeActionInitialization::LXeActionInitialization(LXeRecorderBase* recorder)
 : G4VUserActionInitialization(), fRecorder(recorder)
{}


LXeActionInitialization::~LXeActionInitialization()
{}


void LXeActionInitialization::BuildForMaster() const
{
  SetUserAction(new LXeRunAction(fRecorder));
}


void LXeActionInitialization::Build() const
{
  SetUserAction(new LXePrimaryGeneratorAction());

  SetUserAction(new LXeStackingAction());

  SetUserAction(new LXeRunAction(fRecorder));
  SetUserAction(new LXeEventAction(fRecorder));
  SetUserAction(new LXeTrackingAction(fRecorder));
  SetUserAction(new LXeSteppingAction(fRecorder));
}


G4VSteppingVerbose* LXeActionInitialization::InitializeSteppingVerbose() const
{
  return new LXeSteppingVerbose();
}

