#include "OPTICKS_LOG.hh"

#include "spath.h"
#include "U4GDML.h"
#include "U4Traverse.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* path = spath::Resolve("$HOME/.opticks/GEOM/$GEOM/origin.gdml"); 

    const G4VPhysicalVolume* world = U4GDML::Read(path) ;  

    U4Traverse::Traverse(world); 


    return 0 ;  
}
