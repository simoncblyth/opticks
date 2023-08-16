#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4UnionSolid.hh"
#include "G4RotationMatrix.hh"

#include "U4Solid.h"

#ifdef WITH_SND
#include "snd.hh"
#include "scsg.hh"
#else
#include "sn.h"
#include "s_csg.h"
#endif



void test_Convert(const G4VSolid* solid )
{
    int lvid = 0 ; 
    int depth = 0 ; 
    int level = 1 ; 
#ifdef WITH_SND
    int idx = U4Solid::Convert(solid, lvid, depth, level); 
    std::cout << snd::Desc(idx); 
#else
    sn* nd = U4Solid::Convert(solid, lvid, depth, level); 
    std::cout << nd->desc() ;  
#endif
}

void test_Orb()
{
    G4Orb* orb = new G4Orb("orb", 100.) ; 
    test_Convert(orb); 
}
void test_Box()
{
    G4Box* box = new G4Box("box", 100., 200., 300. ) ; 
    test_Convert(box); 
}
void test_UnionSolid()
{
    G4Orb* orb = new G4Orb("orb", 100.) ; 
    G4Box* box = new G4Box("box", 100., 200., 300. ) ; 

    G4RotationMatrix* rot = new G4RotationMatrix(
            G4ThreeVector(0.707107,-0.707107,0.000000),
            G4ThreeVector(0.707107, 0.707107,0.000000),
            G4ThreeVector(0.000000, 0.000000,1.000000)
            );
    G4ThreeVector tla(50.,60.,70.);
    G4UnionSolid* orb_box = new G4UnionSolid( "orb_box", orb, box, rot, tla );  

    test_Convert(orb_box); 
}

int main(int argc, char** argv)
{
#ifdef WITH_SND
    snd::SetPOOL(new scsg); 
#else
    s_csg* csg = new s_csg ; 
    assert(csg); 
#endif

    test_Orb(); 
    test_Box(); 
    test_UnionSolid(); 

    return 0 ; 
}


