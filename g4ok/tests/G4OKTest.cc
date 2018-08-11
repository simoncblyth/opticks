#include <cassert>
#include "OPTICKS_LOG.hh"
#include "G4Opticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ;

    G4Opticks* om = G4Opticks::GetOpticks() ; 

    assert( om ) ;

    LOG(info) << om->desc() ; 
 

    return 0 ;
}
