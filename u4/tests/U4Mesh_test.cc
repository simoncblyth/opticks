#include "G4Orb.hh"
#include "U4Mesh.h"

int main()
{
    G4Orb* solid = new G4Orb("Orb", 100); 
    G4cout << *solid << std::endl ; 
    NPFold* fold = U4Mesh::Serialize(solid) ; 
    fold->save("$FOLD"); 
    return 0 ; 
};
