#pragma once

#include <map>
#include <string>
#include <glm/glm.hpp>

// ggeo-
class GCache ;
class GGeoTestConfig ; 
class GMaterial ;
class GCSG ; 

// cfg4-
class CMaker ; 
class CPropLib ; 

// g4-
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;
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


class CDetector : public G4VUserDetectorConstruction
{
 public:
    CDetector(GCache* cache, GGeoTestConfig* config);
  private:
    void init();
  public:
    virtual G4VPhysicalVolume* Construct();
    virtual ~CDetector();

  public:
    void setVerbosity(unsigned int verbosity);
    bool isPmtInBox();
    bool isBoxInBox();
    const glm::vec4& getCenterExtent();
    G4VPhysicalVolume* getPV(const char* name);
    CPropLib* getPropLib();


    void dumpPV(const char* msg="CDetector::dumpPV");
  private:
    void makePMT(G4LogicalVolume* mother);
    void kludgePhotoCathode();
    G4LogicalVolume* makeLV(GCSG* csg, unsigned int i);
    void setCenterExtent(float x, float y, float z, float w);

  private:
    GCache*            m_cache ; 
    GGeoTestConfig*    m_config ; 
    CPropLib*          m_lib ; 
    CMaker*            m_maker ; 
    int                m_verbosity ; 

  private:
    glm::vec4          m_center_extent ; 
    std::map<std::string, G4VPhysicalVolume*> m_pvm ; 

};


inline CDetector::CDetector(GCache* cache, GGeoTestConfig* config)
  : 
  m_cache(cache),
  m_config(config),
  m_lib(NULL),
  m_maker(NULL),
  m_verbosity(0)
{
    init();
}


inline void CDetector::setCenterExtent(float x, float y, float z, float w)
{
    m_center_extent.x = x ; 
    m_center_extent.y = y ; 
    m_center_extent.z = z ; 
    m_center_extent.w = w ; 
}

inline const glm::vec4& CDetector::getCenterExtent()
{
    return m_center_extent ; 
}

inline void CDetector::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}

inline CPropLib* CDetector::getPropLib()
{
    return m_lib ; 
}

inline CDetector::~CDetector()
{
}


