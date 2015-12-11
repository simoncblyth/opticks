#include <cstdio>

#include "G4RunManager.hh"

#include "G4UImanager.hh"
#include "G4String.hh"
#include "G4UIExecutive.hh"

#include "LXePhysicsList.hh"
#include "LXeDetectorConstruction.hh"
#include "LXeActionInitialization.hh"
#include "LXeRecorderBase.hh"


int main(int argc, char** argv)
{
    printf("%s\n", argv[0]);

    G4RunManager* runManager = new G4RunManager;
    runManager->SetUserInitialization(new LXeDetectorConstruction());
    runManager->SetUserInitialization(new LXePhysicsList());

    LXeRecorderBase* recorder = NULL; //No recording is done in this example
    runManager->SetUserInitialization(new LXeActionInitialization(recorder));

    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    if(argc==1)
    {
        G4UIExecutive* ui = new G4UIExecutive(argc, argv);
        ui->SessionStart();
        delete ui ; 
    }
    else
    {
        G4String command = "/control/execute ";
        G4String filename = argv[1];
        UImanager->ApplyCommand(command+filename);

    }
    delete runManager;
    return 0 ; 
}
