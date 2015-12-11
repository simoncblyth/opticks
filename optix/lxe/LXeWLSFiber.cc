#include "LXeWLSFiber.hh"
#include "globals.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4SystemOfUnits.hh"

G4LogicalVolume* LXeWLSFiber::fClad2_log=NULL;


LXeWLSFiber::LXeWLSFiber(G4RotationMatrix *pRot,
                             const G4ThreeVector &tlate,
                             G4LogicalVolume *pMotherLogical,
                             G4bool pMany,
                             G4int pCopyNo,
                             LXeDetectorConstruction* c)
  :G4PVPlacement(pRot,tlate,
                 new G4LogicalVolume(new G4Box("temp",1,1,1),
                                     G4Material::GetMaterial("Vacuum"),
                                     "temp",0,0,0),
                 "Cladding2",pMotherLogical,pMany,pCopyNo),fConstructor(c)
{
  CopyValues();

  // The Fiber
  //
  G4Tubs* fiber_tube =
   new G4Tubs("Fiber",fFiber_rmin,fFiber_rmax,fFiber_z,fFiber_sphi,fFiber_ephi);
 
  G4LogicalVolume* fiber_log =
      new G4LogicalVolume(fiber_tube,G4Material::GetMaterial("PMMA"),
                          "Fiber",0,0,0);
 
  // Cladding (first layer)
  //
  G4Tubs* clad1_tube =
      new G4Tubs("Cladding1",fClad1_rmin,fClad1_rmax,fClad1_z,fClad1_sphi,
                 fClad1_ephi);
 
  G4LogicalVolume* clad1_log =
      new G4LogicalVolume(clad1_tube,G4Material::GetMaterial("Pethylene1"),
                          "Cladding1",0,0,0);
 
  // Cladding (second layer)
  //
  G4Tubs* clad2_tube =
      new G4Tubs("Cladding2",fClad2_rmin,fClad2_rmax,fClad2_z,fClad2_sphi,
                 fClad2_ephi);
 
  fClad2_log =
      new G4LogicalVolume(clad2_tube,G4Material::GetMaterial("Pethylene2"),
                          "Cladding2",0,0,0);
 
  new G4PVPlacement(0,G4ThreeVector(0.,0.,0.),fiber_log,
                    "Fiber", clad1_log,false,0);
  new G4PVPlacement(0,G4ThreeVector(0.,0.,0.),clad1_log,
                    "Cladding1",fClad2_log,false,0);

  SetLogicalVolume(fClad2_log);
}


void LXeWLSFiber::CopyValues(){

  fFiber_rmin = 0.00*cm;
  fFiber_rmax = 0.10*cm;    
  fFiber_z    = fConstructor->GetScintX()/2;
  fFiber_sphi = 0.00*deg;
  fFiber_ephi = 360.*deg;

  fClad1_rmin = 0.;// fFiber_rmax;
  fClad1_rmax = fFiber_rmax + 0.015*fFiber_rmax;

  fClad1_z    = fFiber_z;
  fClad1_sphi = fFiber_sphi;
  fClad1_ephi = fFiber_ephi;

  fClad2_rmin = 0.;//fClad1_rmax;
  fClad2_rmax = fClad1_rmax + 0.015*fFiber_rmax;

  fClad2_z    = fFiber_z;
  fClad2_sphi = fFiber_sphi;
  fClad2_ephi = fFiber_ephi;

}
