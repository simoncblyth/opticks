#include "SPath.hh"
#include "U4GDML.h"

int main(int argc, char** argv)
{
    const char* path = SPath::Resolve("$OPTICKS_PREFIX/origin_CGDMLKludge_02mar2022.gdml", NOOP) ; 
    std::cout << " path " << path << std::endl ;    

    G4VPhysicalVolume* world = U4GDML::Parse(path) ;  
    std::cout << " world " << world << std::endl ;    

    return 0 ; 

}
