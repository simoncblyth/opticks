/**
C4Test.cc
===========

The scope "level" of this test main corresponds to 
that of G4CXOpticks::setGeometry as that is where 
this functionality will be used. 

**/

#include "OPTICKS_LOG.hh"

#include "SSim.hh"
#include "U4VolumeMaker.hh"
#include "C4.hh"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
    LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;  
    if(world == nullptr) return 0 ; 

    SSim::Create(); 

    CSGFoundry* fd = C4::Translate(world) ; 

    LOG(info) << " fd " << fd->desc() ; ; 

    return 0 ; 
}
