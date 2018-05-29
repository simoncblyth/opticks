#pragma once

#include "G4VUserDetectorConstruction.hh"
//#include "globals.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    virtual ~DetectorConstruction();
    virtual G4VPhysicalVolume* Construct();
  private:
     G4VPhysicalVolume* ConstructVolume( G4double size, const char* soname, const char* matname, const char* lvn, const char* pvn, G4LogicalVolume* mother );

};


