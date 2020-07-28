/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
class GGeoBase ; 
class GSurfaceLib ; 
class GMaterialLib ; 

// cfg4-
class CBndLib ; 
class CSurfaceLib ; 
class CSensitiveDetector ; 

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
    static const plog::Severity LEVEL ; 
 public:
    friend class CTestDetector ; 
    friend class CGDMLDetector ; 
 public:
    CDetector(OpticksHub* hub, OpticksQuery* query, CSensitiveDetector* sd);
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
    NBoundingBox*      getBoundingBox() const ;
    GSurfaceLib*       getGSurfaceLib() const ;
    CSurfaceLib*       getSurfaceLib() const ;
    GMaterialLib*      getGMaterialLib() const ;
    CMaterialLib*      getMaterialLib() const ;
    G4VPhysicalVolume* getTop() const ;
    bool               isValid() const ;
#ifdef OLD_SENSOR
    void               hookupSD(); 
#endif
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
    OpticksHub*         m_hub ;
    CSensitiveDetector* m_sd ; 
 protected: 
    Opticks*           m_ok ;
    bool               m_dbgsurf ; 
    GGeo*              m_ggeo ; 
    GGeoBase*          m_ggb ; 
    CBndLib*           m_blib ; 
 private:
    OpticksQuery*      m_query ;
    OpticksResource*   m_resource ;
    GMaterialLib*      m_gmateriallib ; 
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

}; 
#include "CFG4_TAIL.hh"


