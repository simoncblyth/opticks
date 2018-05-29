
#include "G4RunManager.hh"

#include "DetectorConstruction.hh"
#include "RunAction.hh"

int main(int argc,char** argv)
{
    G4Random::setTheEngine(new CLHEP::RanecuEngine);

    G4RunManager* runManager = new G4RunManager;

    runManager->SetUserInitialization(new DetectorConstruction());

    runManager->SetUserAction(new RunAction());

    return 0 ; 
}

