#pragma once

class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Box;
class G4Sphere;

class G4Material ; 


#include "G4VUserDetectorConstruction.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:

    DetectorConstruction();
    virtual ~DetectorConstruction();

    virtual G4VPhysicalVolume* Construct();
  private:
    G4Material* MakeWater();
    G4Material* MakeVacuum();
    G4VPhysicalVolume* ConstructDetector();

  private:
    G4LogicalVolume*   m_box_log;
    G4VPhysicalVolume* m_box_phys;
    G4VPhysicalVolume* m_sphere_phys;

    G4Material*        m_vacuum;
    G4Material*        m_water ;


};


inline DetectorConstruction::DetectorConstruction()
  : 
  m_vacuum(NULL),
  m_water(NULL),
  m_box_log(NULL),
  m_box_phys(NULL),
  m_sphere_phys(NULL)
{
}



