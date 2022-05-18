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

    // GGeo machinery, needed when starting from a gdml path or live G4 pv,  
    // still needs Opticks instance,  TODO: avoid this
    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  

    G4CXOpticks gx ;  

    //gx.setGeometry(SPath::SomeGDMLPath()); 
    gx.setGeometry(CSGFoundry::Load()); 

    gx.render_snap();  // sensitive to MOI, EYE, LOOK, UP
 
    return 0 ; 
}
