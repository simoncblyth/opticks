
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "G4Event.hh"
#include "U4VPrimaryGenerator.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt::Create(SEvt::EGPU) ; 

    SEvt::AddTorchGenstep(); 

    return 0 ; 
}
