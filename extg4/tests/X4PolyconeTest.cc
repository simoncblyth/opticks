#include "OPTICKS_LOG.hh"

#include "Opticks.hh"
#include "G4Polycone.hh"
#include "X4Solid.hh"
#include "X4.hh"
#include "NNode.hpp"



G4VSolid* make_Polycone()
{
    G4double phiStart = 0 ; 
    G4double phiTotal = CLHEP::twopi ; 
    G4int numZPlanes = 4 ; 

    double zPlane[] = { 3937, 4000.02, 4000.02, 4094.62 } ;
    double rInner[] = {    0,       0,       0,     0   } ;
    double rOuter[] = { 2040,    2040,    1930,   125   } ;


    G4Polycone* pc = new G4Polycone("poly", phiStart, phiTotal, numZPlanes, zPlane, rInner, rOuter );

    G4VSolid* so = pc ; 

    return so ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 

    G4VSolid* so = make_Polycone() ; 

    std::cout << *so << std::endl ; 

    X4Solid xs(so, &ok, true) ; 

    nnode* root = xs.root(); 

    root->dump_g4code();  

    return 0 ; 
}

