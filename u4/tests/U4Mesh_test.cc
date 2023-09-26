#include "G4Orb.hh"
#include "U4Mesh.h"
int main()
{
    G4Orb* solid = new G4Orb("Orb", 100); 
    U4Mesh::Save(solid, "$FOLD") ; 
    return 0 ; 
};
