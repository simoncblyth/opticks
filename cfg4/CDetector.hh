#pragma once

#include <map>
#include <string>

// okc-
class Opticks ; 
class OpticksHub ; 
class OpticksResource ; 
class OpticksQuery ; 

// gg-
//class GGeo ; 
class GGeoBase ; 
class GSurfaceLib ; 

// cfg4-
class CBndLib ; 
class CSurfaceLib ; 

class CMaterialLib ; 
class CTraverser ; 
class CCheck ; 

// g4-
class G4VPhysicalVolume;

// npy-
template <typename T> class NPY ; 
class NBoundingBox ;

#include <glm/fwd.hpp>
#include "G4VUserDetectorConstruction.hh"
#include "plog/Severity.h"

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
    GSurfaceLib*       getGSurfaceLib() const ;
    CSurfaceLib*       getSurfaceLib() const ;
    CMaterialLib*      getMaterialLib() const ;
    G4VPhysicalVolume* getTop();
    bool               isValid();
 protected:
    void               setValid(bool valid); 
 public: 
    // from traverser : pv and lv are collected into vectors by CTraverser::AncestorVisit
    const G4VPhysicalVolume* getPV(unsigned index);
    const G4VPhysicalVolume* getPV(const char* name);
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
 public: 
    void  export_gdml(const char* dir, const char* name);
    void  export_dae(const char* dir, const char* name);
 private:
    OpticksHub*        m_hub ;
 protected: 
    Opticks*           m_ok ;
    bool               m_dbgsurf ; 
    GGeoBase*          m_ggb ; 
    CBndLib*           m_blib ; 
 private:
    OpticksQuery*      m_query ;
    OpticksResource*   m_resource ;
    CMaterialLib*      m_mlib ; 
    GSurfaceLib*       m_gsurfacelib ; 
    CSurfaceLib*       m_slib ; 
    G4VPhysicalVolume* m_top ;
    CTraverser*        m_traverser ; 
    CCheck*            m_check ; 
    NBoundingBox*      m_bbox ; 
    int                m_verbosity ; 
    bool               m_valid ;
    std::map<std::string, G4VPhysicalVolume*> m_pvm ; 
    std::map<std::string, G4LogicalVolume*>   m_lvm ; 
    plog::Severity                            m_level ; 

}; 
#include "CFG4_TAIL.hh"


