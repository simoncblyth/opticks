#include "OPTICKS_LOG.hh"

#include "U4GDML.h"
#include "U4Traverse.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const G4VPhysicalVolume* world = U4GDML::Read() ;

    LOG_IF(fatal, world == nullptr)
       << " U4GDML::Read FAILED to provide world "
       ;

    if(world == nullptr) return 0;

    U4Traverse::Traverse(world);


    return 0 ;
}
