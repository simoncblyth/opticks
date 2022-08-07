#include "SPath.hh"
#include "SOpticksResource.hh"
#include "OPTICKS_LOG.hh"
#include "U4GDML.h"
#include "U4Material.hh"
#include "NPFold.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* srcpath = argc > 1 ? argv[1] : SOpticksResource::SomeGDMLPath() ; 
    const G4VPhysicalVolume* world = U4GDML::Read(srcpath) ;  

    NPFold* mats = U4Material::GetPropertyFold() ; 

    LOG(info) 
        << " argv[0] " << argv[0] << std::endl 
        << " srcpath " << srcpath << std::endl 
        << " world " << world << std::endl  
        ;

    const char* base = "$TMP/U4Material_GetPropertyFold" ; 
    const char* name = "U4Material" ; 
    const char* fold = SPath::Resolve(base, name, DIRPATH) ; 

    LOG(info) << std::endl << mats->desc_subfold(name) << std::endl ; 

    mats->save(fold); 

    LOG(info) << " fold [" << fold << "]" ; 
    return 0 ; 
}
