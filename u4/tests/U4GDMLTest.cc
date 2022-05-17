#include "SPath.hh"
#include "U4GDML.h"

int main(int argc, char** argv)
{
    const char* path0 = SPath::Resolve("$IDPath/origin.gdml", NOOP ); 
    const char* path1 = SPath::Resolve("$OPTICKS_PREFIX/origin_CGDMLKludge_02mar2022.gdml", NOOP) ; 
    const char* path = SPath::PickFirstExisting(path0, path1); 

    std::cout << " path " << path << std::endl ;    

    G4VPhysicalVolume* world = U4GDML::Read(path) ;  
    std::cout << " world " << world << std::endl ;    

    U4GDML::Write(world, "/tmp/out.gdml"); 

    return 0 ; 

}
