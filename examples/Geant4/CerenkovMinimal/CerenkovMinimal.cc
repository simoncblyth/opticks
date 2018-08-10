#include "OPTICKS_LOG.hh"

#include "G4.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4 g ; 
    g.beamOn(1); 

    return 0 ; 
}


