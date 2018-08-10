#pragma once

#include "G4VUserDetectorConstruction.hh"
#include "G4MaterialPropertyVector.hh"

class G4Material ; 
class G4VPhysicalVolume ;

struct DetectorConstruction : public G4VUserDetectorConstruction
{
    static G4Material* MakeAir(); 
    static G4Material* MakeWater(); 
    static G4MaterialPropertyVector* MakeAirRI() ;
    static G4MaterialPropertyVector* MakeWaterRI() ;
    virtual G4VPhysicalVolume* Construct();
};

