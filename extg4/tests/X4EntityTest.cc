#include <cassert>
#include <cstring>
#include "X4Entity.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    assert( X4Entity::Type("G4Box")    == _G4Box );  
    assert( X4Entity::Type("G4Sphere") == _G4Sphere );  

    assert( strcmp( X4Entity::Name(_G4Box),    "G4Box" ) == 0 );  
    assert( strcmp( X4Entity::Name(_G4Sphere), "G4Sphere" ) == 0 );  


    return 0 ; 
}

