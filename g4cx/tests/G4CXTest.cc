#include "SPath.hh"
#include "G4CXOpticks.hh"

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    // TODO: use GDXML instead of the old CGDMLKludge 
    const char* path0 = SPath::Resolve("$IDPath/origin_CGDMLKludge.gdml", NOOP );  
    const char* path1 = SPath::Resolve("$OPTICKS_PREFIX/origin_CGDMLKludge_02mar2022.gdml", NOOP) ; 
    const char* path = SPath::PickFirstExisting(path0, path1); 
    std::cout << " path " << ( path ? path : "-" ) << std::endl ;    
    assert(path); 

    // GGeo machinery still needs Opticks instance TODO: avoid this
    const char* argforce = "--gparts_transform_offset" ; 
    Opticks ok(argc, argv, argforce); 
    ok.configure(); 

    G4CXOpticks gx ;  
    gx.setGeometry(path); 


    return 0 ; 
}
