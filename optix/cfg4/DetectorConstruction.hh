#pragma once

class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Box;
class G4Sphere;

class G4Material ; 
class G4NistManager ;
class G4MaterialPropertiesTable ; 

#include <glm/glm.hpp>
#include "G4VUserDetectorConstruction.hh"

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    static G4double photonEnergy[] ;

    DetectorConstruction();
    const glm::vec4& getCenterExtent();
    const glm::vec4& getBoundaryDomain();

    virtual ~DetectorConstruction();

    void DumpDomain(const char* msg="DetectorConstruction::DumpDomain");
    virtual G4VPhysicalVolume* Construct();
  private:
    void init();
  private:
    G4VPhysicalVolume* ConstructDetector();
    G4MaterialPropertiesTable* MakeWaterProps();
    G4MaterialPropertiesTable* MakeVacuumProps();



  private:
    G4LogicalVolume*   m_box_log;
    G4VPhysicalVolume* m_box_phys;
    G4VPhysicalVolume* m_sphere_phys;

    G4Material*        m_vacuum;
    G4Material*        m_water ;

    G4NistManager*     m_nistMan ; 

    glm::vec4          m_center_extent ; 
    glm::vec4          m_boundary_domain ; 

};


inline DetectorConstruction::DetectorConstruction()
  : 
  m_vacuum(NULL),
  m_water(NULL),
  m_box_log(NULL),
  m_box_phys(NULL),
  m_sphere_phys(NULL),
  m_nistMan(NULL)
{
    init();
}


inline const glm::vec4& DetectorConstruction::getCenterExtent()
{
    return m_center_extent ; 
}
inline const glm::vec4& DetectorConstruction::getBoundaryDomain()
{
    return m_boundary_domain ; 
}




