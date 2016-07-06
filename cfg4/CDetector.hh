#pragma once

#include <map>
#include <string>

// okc-
class Opticks ; 
class OpticksResource ; 
class OpticksQuery ; 

// cfg4-
class CPropLib ; 
class CTraverser ; 

// g4-
class G4VPhysicalVolume;

// npy-
template <typename T> class NPY ; 
class NBoundingBox ;

#include <glm/fwd.hpp>
#include "G4VUserDetectorConstruction.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CDetector
~~~~~~~~~~

*CDetector* is the base class of *CGDMLDetector* and *CTestDetector*, 
it is a *G4VUserDetectorConstruction* providing detector geometry to Geant4.


**/

class CFG4_API CDetector : public G4VUserDetectorConstruction
{
 public:
    friend class CTestDetector ; 
    friend class CGDMLDetector ; 
 public:
    CDetector(Opticks* opticks, OpticksQuery* query);
    void setVerbosity(unsigned int verbosity);
    virtual ~CDetector();
 private:
    void init();
    void traverse(G4VPhysicalVolume* top); 
 public:
    void setTop(G4VPhysicalVolume* top);   // invokes the traverse
 public:
    // G4VUserDetectorConstruction 
    virtual G4VPhysicalVolume* Construct();
 public: 
    NBoundingBox*      getBoundingBox();
    CPropLib*          getPropLib();
    G4VPhysicalVolume* getTop();
    bool               isValid();
 protected:
    void               setValid(bool valid); 
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
    void saveBuffers(const char* objname, unsigned int objindex);
 public: 
    // via bbox
    const glm::vec4& getCenterExtent();
 private:
    Opticks*           m_opticks ;
    OpticksQuery*      m_query ;
    OpticksResource*   m_resource ;
    CPropLib*          m_lib ; 
    G4VPhysicalVolume* m_top ;
    CTraverser*        m_traverser ; 
    NBoundingBox*      m_bbox ; 
    int                m_verbosity ; 
    bool               m_valid ;
    std::map<std::string, G4VPhysicalVolume*> m_pvm ; 
}; 
#include "CFG4_TAIL.hh"


