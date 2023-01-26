#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4UnionSolid.hh"
#include "G4RotationMatrix.hh"

#include "U4Solid.h"

#include "snd.h"
std::vector<snd> snd::node  = {} ; 
std::vector<spa> snd::param = {} ; 
std::vector<sxf> snd::xform = {} ; 
std::vector<sbb> snd::aabb  = {} ; 
// HMM: how to avoid ? 



void test_Orb()
{
    G4Orb* orb = new G4Orb("orb", 100.) ; 
    int nd = U4Solid::Convert(orb); 
    std::cout << snd::Desc(nd)  ; 
}

void test_Box()
{
    G4Box* box = new G4Box("box", 100., 200., 300. ) ; 
    int nd = U4Solid::Convert(box); 
    std::cout << snd::Desc(nd)  ; 
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
    int nd = U4Solid::Convert(uni); 
    std::cout << snd::Desc(nd)  ; 
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


