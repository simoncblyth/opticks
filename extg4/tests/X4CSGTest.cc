#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "X4CSG.hh"

// start of portion to be generated ----------------
#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"
#include "G4UnionSolid.hh"

G4VSolid* make_solid()
{
    G4VSolid* b = new G4Orb("orb",10) ;
    G4VSolid* d = new G4Box("box",7,7,7) ;
    G4RotationMatrix* A = new G4RotationMatrix(G4ThreeVector(0.707107,-0.707107,0.000000),G4ThreeVector(0.707107,0.707107,0.000000),G4ThreeVector(0.000000,0.000000,1.000000));
    G4ThreeVector B(1.000000,0.000000,0.000000);
    G4VSolid* a = new G4UnionSolid("uni1",b , d , A , B) ;
    return a ; 
}
// end of portion to be generated ---------------------


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure(); 

    G4VSolid* solid = make_solid() ; 

    //const char* csgpath = "$TMP/X4CSGTest" ; 
    //X4CSG::Serialize( solid, csgpath ) ;
     
    const char* prefix = "$TMP/x4gen/tests" ; 
    unsigned lvidx = 1 ; 
    X4CSG::GenerateTest( solid, &ok, prefix, lvidx ) ;

    return 0 ; 
}


