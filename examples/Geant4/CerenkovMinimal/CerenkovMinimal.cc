#include "OPTICKS_LOG.hh"
#include "G4.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    int nevt = argc > 1 ? atoi(argv[1]) : 3 ; 
    G4 g(nevt) ; 
    return 0 ; 
}

