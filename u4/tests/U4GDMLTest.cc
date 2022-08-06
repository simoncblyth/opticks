#include "SPath.hh"
#include "SOpticksResource.hh"
#include "OPTICKS_LOG.hh"
#include "U4GDML.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* srcpath = SOpticksResource::SomeGDMLPath(); 
    const char* dstpath = SPath::Resolve("$TMP/U4GDMLTest/out.gdml", FILEPATH) ; 

    const G4VPhysicalVolume* world = U4GDML::Read(srcpath) ;  

    U4GDML::Write(world, dstpath ); 

    std::cout 
        << " srcpath " << srcpath << std::endl 
        << " dstpath " << dstpath << std::endl 
        << " world " << world << std::endl  
        ;

    return 0 ; 

}
