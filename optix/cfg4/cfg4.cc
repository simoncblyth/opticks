#include <cstdio>

#include "G4RunManager.hh"

#include "G4UImanager.hh"
#include "G4String.hh"
#include "G4UIExecutive.hh"

#include "PhysicsList.hh"
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"
#include "Recorder.hh"


int main(int argc, char** argv)
{
    printf("%s\n", argv[0]);

    G4RunManager* runManager = new G4RunManager;
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserInitialization(new DetectorConstruction());

    RecorderBase* recorder = new Recorder(1000,10); // photon_max, step_max
    runManager->SetUserInitialization(new ActionInitialization(recorder));
    runManager->Initialize();
    runManager->BeamOn(1);

    recorder->save("/tmp/recorder.npy");

    delete runManager;
    return 0 ; 
}
