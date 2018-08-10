#include "OPTICKS_LOG.hh"

// steering 
#include "G4RunManager.hh"
#include "G4UIManager.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "PrimaryGeneratorAction.hh"
#include "SteppingAction.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4RunManager* rm = new G4RunManager;
    rm->SetUserInitialization(new DetectorConstruction());
    rm->SetUserInitialization(new PhysicsList());
    rm->SetUserAction(new PrimaryGeneratorAction()); 
    rm->Initialize(); 
    rm->BeamOn(10); 

    return 0 ; 
}


