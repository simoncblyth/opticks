#pragma once

#include <map>
#include <string>

// okc-
class Opticks ; 
class OpticksHub ; 
class OpticksResource ; 
class OpticksQuery ; 

// gg-
class GGeo ; 
class GSurLib ; 

// cfg4-
class CBndLib ; 
class CSurLib ; 
class CMaterialLib ; 
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
===========

*CDetector* is the base class of *CGDMLDetector* and *CTestDetector*, 
it is a *G4VUserDetectorConstruction* providing detector geometry to Geant4.
The *CDetector* instance is a constituent of *CGeometry* which is 
instanciated in *CGeometry::init*.


**/

class CFG4_API CDetector : public G4VUserDetectorConstruction
{
 public:
    friend class CTestDetector ; 
    friend class CGDMLDetector ; 
 public:
    CDetector(OpticksHub* hub, OpticksQuery* query);
    void setVerbosity(unsigned int verbosity);
    void attachSurfaces();
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
    CMaterialLib*      getPropLib();
    G4VPhysicalVolume* getTop();
    bool               isValid();
 protected:
    void               setValid(bool valid); 
 public: 
    // from traverser
    const G4VPhysicalVolume* getPV(unsigned index);
    const G4LogicalVolume*   getLV(unsigned index);
    const G4LogicalVolume*   getLV(const char* name);

    void dumpLV(const char* msg="CDetector::dumpLV");
 public: 
    const char*    getPVName(unsigned int index);
    // from local m_pvm map used for CTestDetector, TODO: adopt traverser for this
    G4VPhysicalVolume* getLocalPV(const char* name);
    void dumpLocalPV(const char* msg="CDetector::dumpLocalPV");

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
    OpticksHub*        m_hub ;
 protected: 
    Opticks*           m_ok ;
    GGeo*              m_ggeo ; 
    CBndLib*           m_blib ; 
    GSurLib*           m_gsurlib ; 
    CSurLib*           m_csurlib ; 
 private:
    OpticksQuery*      m_query ;
    OpticksResource*   m_resource ;
    CMaterialLib*      m_mlib ; 
    G4VPhysicalVolume* m_top ;
    CTraverser*        m_traverser ; 
    NBoundingBox*      m_bbox ; 
    int                m_verbosity ; 
    bool               m_valid ;
    std::map<std::string, G4VPhysicalVolume*> m_pvm ; 
    std::map<std::string, G4LogicalVolume*>   m_lvm ; 
}; 
#include "CFG4_TAIL.hh"


