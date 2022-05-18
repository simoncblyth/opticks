#include "SPath.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  // GGeo machinery still needs Opticks instance,  TODO: avoid this

    G4CXOpticks gx ;  
    gx.setGeometry(SPath::SomGDMLPath()); 

    return 0 ; 
}
