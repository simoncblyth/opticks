
#include "OPTICKS_LOG.hh"

#include "U4VolumeMaker.hh"
#include "C4.hh"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
    LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;  
    if(world == nullptr) return 0 ; 

    CSGFoundry* fd = C4::Translate(world) ; 

    LOG(info) << " fd " << fd->desc() ; ; 

    return 0 ; 
}
