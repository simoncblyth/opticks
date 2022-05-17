#include "SPath.hh"
#include "G4CXOpticks.hh"

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    // GGeo machinery still needs Opticks instance TODO: avoid this
    const char* argforce = "--gparts_transform_offset" ; 
    Opticks ok(argc, argv, argforce); 
    ok.configure(); 


    G4CXOpticks gx ;  
    const char* path = SPath::Resolve("$OPTICKS_PREFIX/origin_CGDMLKludge_02mar2022.gdml", NOOP) ; 
    gx.setGeometry(path); 


    return 0 ; 
}
