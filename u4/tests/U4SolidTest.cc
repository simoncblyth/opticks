#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4UnionSolid.hh"
#include "G4RotationMatrix.hh"

#include "U4Solid.h"

void test_Orb()
{
    G4Orb* orb = new G4Orb("orb", 100.) ; 
    snd* nd = U4Solid::Convert(orb); 
    if(nd) std::cout << *nd ; 
}

void test_Box()
{
    G4Box* box = new G4Box("box", 100., 200., 300. ) ; 
    snd* nd = U4Solid::Convert(box); 
    if(nd) std::cout << *nd ; 
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

    G4UnionSolid* uni = new G4UnionSolid( "orb_box", orb, box, rot, tla );  
    snd* nd = U4Solid::Convert(uni); 
    if(nd) std::cout << *nd ; 
}

int main(int argc, char** argv)
{
    /*
    test_Orb(); 
    test_Box(); 
    */

    test_UnionSolid(); 


    return 0 ; 
}


