#pragma once

#include <map>
#include <string>

// ggeo-
class GCache ;

// cfg4-
class CPropLib ; 
class CTraverser ; 

// g4-
class G4VPhysicalVolume;

// npy-
template <typename T> class NPY ; 


#include <glm/glm.hpp>
#include "G4VUserDetectorConstruction.hh"

class CDetector : public G4VUserDetectorConstruction
{
 public:
    friend class CTestDetector ; 
    friend class CGDMLDetector ; 
 public:
    CDetector(GCache* cache);
    virtual ~CDetector();
 private:
    void init();
 public:
    void setVerbosity(unsigned int verbosity);
    void setTop(G4VPhysicalVolume* top);   // invokes the traverse
    virtual G4VPhysicalVolume* Construct();
    void saveTransforms(const char* objname, unsigned int objindex);
 public: 
    unsigned int getNumGlobalTransforms();
    unsigned int getNumLocalTransforms();
    glm::mat4 getGlobalTransform(unsigned int index);
    glm::mat4 getLocalTransform(unsigned int index);
    NPY<float>* getGlobalTransforms();
    NPY<float>* getLocalTransforms();
    const char* getPVName(unsigned int index);
 private:
    void traverse(G4VPhysicalVolume* top); 
 public:
    CPropLib* getPropLib();
    const glm::vec4& getCenterExtent();
    void setCenterExtent(float x, float y, float z, float w);
    G4VPhysicalVolume* getPV(const char* name);
    void dumpPV(const char* msg="CDetector::dumpPV");

  private:
    GCache*            m_cache ; 
    CPropLib*          m_lib ; 
    G4VPhysicalVolume* m_top ;
    CTraverser*        m_traverser ; 
    glm::vec4          m_center_extent ; 
    int                m_verbosity ; 
  private:
    std::map<std::string, G4VPhysicalVolume*> m_pvm ; 




}; 


inline CDetector::CDetector(GCache* cache)
  : 
  m_cache(cache),
  m_lib(NULL),
  m_top(NULL),
  m_traverser(NULL),
  m_verbosity(0)
{
    init();
}

inline G4VPhysicalVolume* CDetector::Construct()
{
    return m_top ; 
}

inline void CDetector::setTop(G4VPhysicalVolume* top)
{
    m_top = top ; 
    traverse(m_top);
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



