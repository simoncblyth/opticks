#include <cassert>

#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"

#include "DetectorConstruction.hh"


DetectorConstruction::DetectorConstruction()
  :
   G4VUserDetectorConstruction()
{ 
}

DetectorConstruction::~DetectorConstruction()
{ 
}

G4VPhysicalVolume* DetectorConstruction::ConstructVolume( G4double size, const char* soname, const char* matname, const char* lvn, const char* pvn, G4LogicalVolume* mother )
{
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* material  = nist->FindOrBuildMaterial(matname);
    G4VSolid* solid = NULL ; 
    if(strcmp(soname, "box") == 0) 
    {
        G4double side = size ; 
        solid = new G4Box("box_solid", side, side, side);     
    }
    else if(strcmp(soname, "sphere") == 0) 
    {
        G4double radius = size ; 
        solid = new G4Sphere("sphere_solid",  0., radius, 0., CLHEP::twopi, 0., CLHEP::pi);
    }
    G4LogicalVolume* lv = new G4LogicalVolume(solid, material, lvn);          
    G4VPhysicalVolume* pv = new G4PVPlacement(0, G4ThreeVector(), lv, pvn, mother, 0, true );
    return pv ;
} 

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4VPhysicalVolume* container = ConstructVolume( 1000*mm, "box", "G4_AIR", "container_lv", "container_pv", NULL );
    G4VPhysicalVolume* object    = ConstructVolume(   500*mm, "sphere", "G4_WATER", "object_lv", "object_pv", container->GetLogicalVolume() );
    assert( object );
    return container ; 
}


