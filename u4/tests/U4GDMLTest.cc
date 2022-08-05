#include "SPath.hh"
#include "SOpticksResource.hh"
#include "U4GDML.h"

int main(int argc, char** argv)
{
    const char* ipath = SOpticksResource::SomeGDMLPath(); 
    const char* opath = SPath::Resolve("$TMP/U4GDMLTest/out.gdml", FILEPATH) ; 

    const G4VPhysicalVolume* world = U4GDML::Read(ipath) ;  

    U4GDML::Write(world, opath ); 

    std::cout 
        << " ipath " << ipath << std::endl 
        << " opath " << opath << std::endl 
        << " world " << world << std::endl  
        ;

    return 0 ; 

}
