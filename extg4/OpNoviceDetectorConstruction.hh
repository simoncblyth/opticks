#pragma once
/**
OpNoviceDetectorConstruction 
=============================

OpNoviceDetectorConstruction is used by X4Sample for the creation of simple geometries
that are used for testing.

Due to Geant4 API changes circa September 2021 the former setting 
of spline interpolation for some properties has been removed. 

**/

#include "globals.hh"
#include "G4VUserDetectorConstruction.hh"


#include "X4_API_EXPORT.hh"

class X4_API OpNoviceDetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    OpNoviceDetectorConstruction();
    virtual ~OpNoviceDetectorConstruction();

  public:
    virtual G4VPhysicalVolume* Construct();

  private:
    G4double fExpHall_x;
    G4double fExpHall_y;
    G4double fExpHall_z;

    G4double fTank_x;
    G4double fTank_y;
    G4double fTank_z;

    G4double fBubble_x;
    G4double fBubble_y;
    G4double fBubble_z;
};


