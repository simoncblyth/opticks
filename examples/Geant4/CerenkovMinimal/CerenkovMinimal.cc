#include "OPTICKS_LOG.hh"
#include "G4.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    int nevt = argc > 1 ? atoi(argv[1]) : 3 ; 
    int opticksMode = argc > 2 ? atoi(argv[2]) : 2 ;   // aim: 1:OK, 2:G4, 3:OK+G4 
    G4 g(nevt, opticksMode) ; 
    return 0 ; 
}

