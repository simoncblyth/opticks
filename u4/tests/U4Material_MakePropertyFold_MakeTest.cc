#include "spath.h"

#include "OPTICKS_LOG.hh"
#include "U4GDML.h"
#include "U4Material.hh"
#include "NPFold.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* _origin = spath::Resolve("$HOME/.opticks/GEOM/$GEOM/origin.gdml"); 
    const char* srcpath = argc > 1 ? argv[1] : spath::Resolve(_origin) ; 

    const G4VPhysicalVolume* world = U4GDML::Read(srcpath) ;  

    NPFold* mats = U4Material::MakePropertyFold() ; 

    LOG(info) 
        << " argv[0] " << argv[0] << std::endl 
        << " srcpath " << srcpath << std::endl 
        << " world " << world << std::endl  
        ;

    const char* base = "$TMP/U4Material_MakePropertyFold" ; 
    const char* name = "U4Material" ; 
    const char* fold = spath::Resolve(base, name) ; 

    LOG(info) << std::endl << mats->desc_subfold(name) << std::endl ; 

    mats->save(fold); 

    LOG(info) << " fold [" << fold << "]" ; 
    return 0 ; 
}
