#include "X4SolidExtent.hh"
#include "G4VSolid.hh"
#include "G4Orb.hh"

#include "NBBox.hpp"
#include "NGLM.hpp"

#include "OPTICKS_LOG.hh"

void test_Extent()
{
    G4Orb* orb = new G4Orb("orb", 100); 
    nbbox* bb = X4SolidExtent::Extent(orb); 
    bb->dump();
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_Extent(); 
    
    return 0 ; 
}


