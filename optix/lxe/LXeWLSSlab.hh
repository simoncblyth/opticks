#ifndef LXeWLSSlab_H
#define LXeWLSSlab_H 1

#include "G4PVPlacement.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4OpticalSurface.hh"

#include "LXeDetectorConstruction.hh"

class LXeWLSSlab : public G4PVPlacement
{
  public:

    LXeWLSSlab(G4RotationMatrix *pRot,
               const G4ThreeVector &tlate,
               G4LogicalVolume *pMotherLogical,
               G4bool pMany,
               G4int pCopyNo,
               LXeDetectorConstruction* c);

  private:

    void CopyValues();

    LXeDetectorConstruction* fConstructor;

    static G4LogicalVolume* fScintSlab_log;

    G4int fNfibers;
    G4double fScint_x;
    G4double fScint_y;
    G4double fScint_z;
    G4double fSlab_z;
};

#endif
