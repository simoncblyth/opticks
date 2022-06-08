
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "G4Event.hh"
#include "U4VPrimaryGenerator.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt evt ; 
    SEvt::AddTorchGenstep(); 

    G4Event* event = new G4Event ; 
    U4VPrimaryGenerator::GeneratePrimaryVertex(event); 

  
    return 0 ; 
}
