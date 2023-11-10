#include "OPTICKS_LOG.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info) << G4CXOpticks::Desc() ; 
    return 0 ; 
}

