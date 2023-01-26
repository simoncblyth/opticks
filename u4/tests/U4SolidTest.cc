#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4UnionSolid.hh"
#include "G4RotationMatrix.hh"

#include "U4Solid.h"
#include "snd.hh"
#include "scsg.hh"

int test_Orb()
{
    G4Orb* orb = new G4Orb("orb", 100.) ; 
    return U4Solid::Convert(orb); 
}
int test_Box()
{
    G4Box* box = new G4Box("box", 100., 200., 300. ) ; 
    return U4Solid::Convert(box); 
}
int test_UnionSolid()
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

    return U4Solid::Convert(uni); 
}

int main(int argc, char** argv)
{
    snd::SetPOOL(new scsg); 

    int idx = -1 ; 

    idx = test_Orb(); 
    std::cout << snd::DescND(idx); 

    idx = test_Box(); 
    std::cout << snd::DescND(idx); 

    idx = test_UnionSolid(); 
    std::cout << snd::DescND(idx); 


    return 0 ; 
}


