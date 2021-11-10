#include "G4Orb.hh"
#include "X4Intersect.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4VSolid* solid = new G4Orb( "orb", 100. ); 

    X4Intersect isect(solid); 
    isect.scan(); 
    //isect.save(); 

    return 0 ; 
}


