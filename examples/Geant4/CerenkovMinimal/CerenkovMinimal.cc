#include "OPTICKS_LOG.hh"
#include "G4.hh"
#include "CMixMaxRng.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CMixMaxRng mmr;  // switch engine to a instrumented shim, to see the random stream

    G4 g(1) ; 
    return 0 ; 
}


