/**
U4NistManagerTest.cc
=====================

::

    G4_WATER
    G4_AIR
    G4_CONCRETE
    G4_Pb

::

    cd /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/examples
    find . -type f -exec grep -H FindOrBuildMaterial {} \;


**/

#include "U4NistManager.h"

int main(int argc, char** argv)
{
    const char* name = argc > 1 ? argv[1] : "G4_WATER" ; 
    std::cout << ( name ? name : "-" ) << std::endl ; 

    G4Material* material  = U4NistManager::GetMaterial(name) ; 

    if(material) G4cout << *material ; 
    else G4cout << argv[0] << " FAILED TO CREATE " << name ; 

    return 0 ;
}
