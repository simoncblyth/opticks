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

class G4Material ; 
class G4MaterialPropertiesTable ; 

#include <string>
#include <glm/glm.hpp>
#include "G4VUserDetectorConstruction.hh"

class Detector : public G4VUserDetectorConstruction
{
  public:
    static std::string PVName(const char* shapename);
    static std::string LVName(const char* shapename);
  public:
    Detector(GCache* cache, GGeoTestConfig* config);
    virtual G4VPhysicalVolume* Construct();
    virtual ~Detector();
  public:
    const glm::vec4& getCenterExtent();
    const glm::vec4& getBoundaryDomain();
    G4VPhysicalVolume* CreateBoxInBox();
  private:
    void init();
  private:
    G4Material* makeVacuum(const char* name);
    G4Material* makeWater(const char* name);
    G4MaterialPropertiesTable* makeMaterialPropertiesTable(unsigned int index);
    G4Material* makeOuterMaterial(const char* spec);
    G4Material* makeInnerMaterial(const char* spec);
    G4Material* makeMaterial(unsigned int mat);
  private:
    void setCenterExtent(float x, float y, float z, float w);
    G4VSolid* makeSolid(char shapecode, const glm::vec4& param);
    G4VSolid* makeBox(const glm::vec4& param);
    G4VSolid* makeSphere(const glm::vec4& param);
  private:
    GCache*            m_cache ; 
    GGeoTestConfig*    m_config ; 
    GBndLib*           m_bndlib ; 
    GMaterialLib*      m_mlib ; 
    GSurfaceLib*       m_slib ; 
  private:
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
{
    init();
}


inline void Detector::setCenterExtent(float x, float y, float z, float w)
{
    m_center_extent.x = x ; 
    m_center_extent.y = y ; 
    m_center_extent.z = z ; 
    m_center_extent.w = w ; 
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


