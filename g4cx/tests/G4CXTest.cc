#include "SPath.hh"
#include "SEventConfig.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "CSGFoundry.h"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEventConfig::SetRGMode("render");  

    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  // GGeo machinery still needs Opticks instance,  TODO: avoid this

    G4CXOpticks gx ;  

    //gx.setGeometry(SPath::SomeGDMLPath()); 
    gx.setGeometry(CSGFoundry::Load()); 

    gx.snap(); 
 
    return 0 ; 
}
