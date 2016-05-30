#pragma once

#include <map>
#include <string>

// optickscore-
class OpticksResource ; 

// ggeo-
class GCache ;

// cfg4-
class CPropLib ; 
class CTraverser ; 

// g4-
class G4VPhysicalVolume;

// npy-
template <typename T> class NPY ; 
class NBoundingBox ;


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
    void saveBuffers(const char* objname, unsigned int objindex);
 public: 
    NBoundingBox*  getBoundingBox();
    CPropLib*      getPropLib();
 public: 
    const char*    getPVName(unsigned int index);
    G4VPhysicalVolume* getPV(const char* name);
    void dumpPV(const char* msg="CDetector::dumpPV");
 public: 
     // via traverser
    unsigned int   getNumGlobalTransforms();
    unsigned int   getNumLocalTransforms();
    glm::mat4      getGlobalTransform(unsigned int index);
    glm::mat4      getLocalTransform(unsigned int index);
    NPY<float>*    getGlobalTransforms();
    NPY<float>*    getLocalTransforms();
 public: 
    // via bbox
    const glm::vec4& getCenterExtent();

  private:
    void traverse(G4VPhysicalVolume* top); 

  private:
    GCache*            m_cache ; 
    OpticksResource*   m_resource ;
    CPropLib*          m_lib ; 
    G4VPhysicalVolume* m_top ;
    CTraverser*        m_traverser ; 
    NBoundingBox*      m_bbox ; 
    int                m_verbosity ; 
    std::map<std::string, G4VPhysicalVolume*> m_pvm ; 
}; 

inline CDetector::CDetector(GCache* cache)
  : 
  m_cache(cache),
  m_resource(NULL),
  m_lib(NULL),
  m_top(NULL),
  m_traverser(NULL),
  m_bbox(NULL),
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
inline void CDetector::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}
inline CPropLib* CDetector::getPropLib()
{
    return m_lib ; 
}
inline NBoundingBox* CDetector::getBoundingBox()
{
    return m_bbox ; 
}

