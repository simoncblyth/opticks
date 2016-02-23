#pragma once

// ggeo-
class GCache ;
class GBndLib ;
class GMaterialLib ;
class GSurfaceLib ;
class GGeoTestConfig ; 
class GMaterial ;
class GCSG ; 

// g4-
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;

class G4Material ; 
class G4MaterialPropertiesTable ; 

#include <string>
#include <glm/glm.hpp>
#include "G4VUserDetectorConstruction.hh"

//
// Hmm for fully controlled cfg4- testing need 
// to construct composite boolean G4 geometries for things like PMTs
//  
// Maybe GParts specs not appropriate level to start from for that ? 
// Need a higher level ? detdesc ?
//
// Parsing detdesc is a shortterm DYB specific kludge, the analytic geometry info 
// should eventually live withing the G4DAE exported file (GDML style). 
// Given this, the approach for making G4 PMTs can be a kludge too. 
// Does not need to be very general, probably will only every use for PMTs,
// as this approach too much effort for full geometries.
//  
// See:  
//       pmt-ecd
//           python detdesc parsing into parts buffer
//
//       GMaker::makeZSphereIntersect    
//           converts high level params (two sphere radii and z offsets) of convex lens
//           into parts  
//
//  Possible approach:
//       during the pmt- python partitioning to create the parts buffer 
//       write a sidecar buffer of high level params, that can be used for G4 boolean CSG
//       


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

    G4VPhysicalVolume* Create();
    void makePMT(G4LogicalVolume* mother);
    bool isPmtInBox();
    bool isBoxInBox();

    G4LogicalVolume* makeLV(GCSG* csg, unsigned int i);
    G4VSolid* makeSolid(GCSG* csg, unsigned int i);


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
    G4MaterialPropertiesTable* makeMaterialPropertiesTable(GMaterial* kmat);
    G4Material* makeMaterial(GMaterial* kmat);
    G4Material* convertMaterial(GMaterial* kmat);
  private:
    void setCenterExtent(float x, float y, float z, float w);
    G4VSolid* makeSolid(GCSG* csg, unsigned int first, unsigned int last);
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


inline Detector::~Detector()
{
}


