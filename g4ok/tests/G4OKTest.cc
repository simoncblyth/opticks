#include <cassert>
#include "OPTICKS_LOG.hh"
#include "G4OpticksManager.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ;

    OPTICKS_LOG_::Check();

    G4OpticksManager* om = G4OpticksManager::GetOpticksManager() ; 

    assert( om ) ;

    LOG(info) << om->desc() ; 
 

    return 0 ;
}
