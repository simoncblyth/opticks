#pragma once

// ggeo-
class GCache ;
class GBndLib ;
class GMaterialLib ;
class GSurfaceLib ;
class GGeoTestConfig ; 

// g4-
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;

//class G4Box;
//class G4Sphere;

class G4Material ; 
//class G4NistManager ;
class G4MaterialPropertiesTable ; 

#include <string>
#include <glm/glm.hpp>
#include "G4VUserDetectorConstruction.hh"

class Detector : public G4VUserDetectorConstruction
{
  public:
    static std::string PVName(const char* shapename);
    static std::string LVName(const char* shapename);
    //static G4double photonEnergy[] ;

  public:
    Detector(GCache* cache, GGeoTestConfig* config);
    virtual ~Detector();

    const glm::vec4& getCenterExtent();
    const glm::vec4& getBoundaryDomain();

    G4VPhysicalVolume* CreateBoxInBox();

    //void DumpDomain(const char* msg="Detector::DumpDomain");
    //virtual G4VPhysicalVolume* Construct_Old();
    virtual G4VPhysicalVolume* Construct();
  private:
    void init();
  private:
    //G4VPhysicalVolume* ConstructDetector_Old();
    //G4MaterialPropertiesTable* MakeWaterProps();
    //G4MaterialPropertiesTable* MakeVacuumProps();

    G4Material* makeVacuum(const char* name);
    G4Material* makeWater(const char* name);

    G4MaterialPropertiesTable* makeMaterialPropertiesTable(unsigned int index);
    G4VSolid* makeSolid(char shapecode, const glm::vec4& param);
    G4VSolid* makeBox(const glm::vec4& param);
    G4VSolid* makeSphere(const glm::vec4& param);

    G4Material* makeOuterMaterial(const char* spec);
    G4Material* makeInnerMaterial(const char* spec);
    G4Material* makeMaterial(unsigned int mat);


  private:
    GCache*            m_cache ; 
    GGeoTestConfig*    m_config ; 
    GBndLib*           m_bndlib ; 
    GMaterialLib*      m_mlib ; 
    GSurfaceLib*       m_slib ; 

  private:
    //G4LogicalVolume*   m_box_log;
    //G4VPhysicalVolume* m_box_phys;
    //G4VPhysicalVolume* m_sphere_phys;
    //G4Material*        m_vacuum;
    //G4Material*        m_water ;
    //G4NistManager*     m_nistMan ; 

    glm::vec4          m_center_extent ; 
    glm::vec4          m_boundary_domain ; 

};


inline Detector::Detector(GCache* cache, GGeoTestConfig* config)
  : 
  m_cache(cache),
  m_config(config),
  m_bndlib(NULL),
  m_mlib(NULL),
  m_slib(NULL)
//  m_vacuum(NULL),
//  m_water(NULL),
//  m_box_log(NULL),
//  m_box_phys(NULL),
//  m_sphere_phys(NULL),
//  m_nistMan(NULL)
{
    init();
}


inline const glm::vec4& Detector::getCenterExtent()
{
    return m_center_extent ; 
}
inline const glm::vec4& Detector::getBoundaryDomain()
{
    return m_boundary_domain ; 
}

inline Detector::~Detector()
{
}


