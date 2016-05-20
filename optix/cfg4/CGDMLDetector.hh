// op --cgdmldetector
#pragma once

#include <map>
#include <string>
#include <glm/glm.hpp>

// ggeo-
class GCache ;

// cfg4-
class CPropLib ; 
class CTraverser ; 

// g4-
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;

#include "G4VUserDetectorConstruction.hh"

class CGDMLDetector : public G4VUserDetectorConstruction
{
  public:
    CGDMLDetector(GCache* cache);
  private:
    void init();
  private:
    void fixMaterials(G4VPhysicalVolume* top);
    void addMPT();
  public:
    virtual G4VPhysicalVolume* Construct();
    virtual ~CGDMLDetector();
  public:
    void setVerbosity(unsigned int verbosity);
  private:
    GCache*            m_cache ; 
    CPropLib*          m_lib ; 
    CTraverser*        m_traverser ; 
    G4VPhysicalVolume* m_top ;
    int                m_verbosity ; 

};

inline CGDMLDetector::CGDMLDetector(GCache* cache)
  : 
  m_cache(cache),
  m_lib(NULL), 
  m_traverser(NULL),
  m_top(NULL), 
  m_verbosity(0)
{
    init();
}

inline void CGDMLDetector::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}

inline CGDMLDetector::~CGDMLDetector()
{
}


