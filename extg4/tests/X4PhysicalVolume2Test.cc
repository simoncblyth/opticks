#include "G4Orb.hh"
#include "G4Box.hh"
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "GMaterialLib.hh"

#include "X4PhysicalVolume.hh"
#include "X4MaterialTable.hh"
#include "OpNoviceDetectorConstruction.hh"
#include "LXe_Materials.hh"
#include "Opticks.hh"
#include "SDirect.hh"
#include "OPTICKS_LOG.hh"


/**


check_transforms
------------------

Checking the glTF in GLTFSceneKitSample.  

* two Orbs appear horizontally left-right with the smaller to the right, 
  x-axis to right.

* making the world thinner in Z, shows +Z is out the screen


        Y 
        |
        |
        | 
        +-----> X
       /
      /
     Z


**/

G4VPhysicalVolume* check_transforms(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   
    G4VSolid* so_0 = new G4Box("World",1.,1.,0.1) ; 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);

    G4LogicalVolume* mo_1 = lv_0 ;   
    G4VSolid* so_1 = new G4Orb("Orb",0.3) ; 
    G4LogicalVolume* lv_1 = new G4LogicalVolume(so_1,lm.fAir,"Orb",0,0,0);
    G4VPhysicalVolume* pv_1 = new G4PVPlacement(0,G4ThreeVector(0.3,0.1,0),lv_1, "Orb",mo_1,false,0);
    assert( pv_1 );

    G4LogicalVolume* mo_2 = lv_0 ;   
    G4VSolid* so_2 = new G4Orb("Orb",0.4) ; 
    G4LogicalVolume* lv_2 = new G4LogicalVolume(so_2,lm.fAir,"Orb",0,0,0);
    G4VPhysicalVolume* pv_2 = new G4PVPlacement(0,G4ThreeVector(-0.3,0,0),lv_2, "Orb",mo_2,false,0);
    assert( pv_2 );

    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}

G4VPhysicalVolume* check_subtraction(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   

    G4VSolid* box = new G4Box("Box",1.,1.,0.1) ; 
    G4VSolid* orb = new G4Orb("Orb",0.3) ; 

    G4VSolid* so_0 = new G4SubtractionSolid("box-orb", box, orb ); 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}

G4VPhysicalVolume* check_subtraction_displaced(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   

    G4VSolid* box = new G4Box("Box",1.,1.,0.1) ; 
    G4VSolid* orb = new G4Orb("Orb",0.3) ; 

    G4RotationMatrix* rotMatrix = NULL ; 
    G4ThreeVector transVector(0.5,0,0);

    G4VSolid* so_0 = new G4SubtractionSolid("box-orb", box, orb, rotMatrix, transVector ); 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}

G4VPhysicalVolume* check_union(const LXe_Materials& lm)
{
    G4LogicalVolume* mo_0 = NULL ;   

    G4VSolid* box = new G4Box("Box",1.,1.,0.1) ; 
    G4VSolid* orb = new G4Orb("Orb",0.3) ; 

    G4VSolid* so_0 = new G4UnionSolid("box+orb", box, orb ); 
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,lm.fAir,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0, "World",mo_0,false,0);
    G4VPhysicalVolume* top = pv_0 ; 
    return top ;  
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LXe_Materials lm ; 

    //G4VPhysicalVolume* top = check_transforms(lm);
    //G4VPhysicalVolume* top = check_subtraction(lm);
    //G4VPhysicalVolume* top = check_union(lm);
    G4VPhysicalVolume* top = check_subtraction_displaced(lm);

    GGeo* ggeo = X4PhysicalVolume::Convert(top) ;   
    assert(ggeo);  

    Opticks* ok = Opticks::GetOpticks();
    ok->Summary();

    return 0 ; 
}


