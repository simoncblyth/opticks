// op --cgdmldetector
#pragma once

// ggeo-
class GCache ;

// cfg4-
class CTraverser ; 

#include "CDetector.hh"
class CGDMLDetector : public CDetector
{
  public:
    CGDMLDetector(GCache* cache);
  private:
    void init();
  private:
    void fixMaterials(G4VPhysicalVolume* top);
    void addMPT();
  public:
    virtual ~CGDMLDetector();
  private:
    CTraverser*        m_traverser ; 
    int                m_verbosity ; 
};

inline CGDMLDetector::CGDMLDetector(GCache* cache)
  : 
  CDetector(cache),
  m_traverser(NULL)
{
    init();
}

inline CGDMLDetector::~CGDMLDetector()
{
}


