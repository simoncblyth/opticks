#include <cassert>
#include "OPTICKS_LOG.hh"
#include "G4OpticksManager.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv) ;

    OPTICKS_LOG::Check();

    G4OpticksManager* ok = G4OpticksManager::GetOpticksManager() ; 

    assert( ok ) ; 

    return 0 ;
}
