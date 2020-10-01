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

#include <vector>

struct NLODConfig ; 
class NCSG ; 
class NCSGList ; 
class Opticks ; 
class OpticksResource ; 
class OpticksEvent ; 

class NGeoTestConfig ; 
class GGeoBase ; 

class GGeoLib ; 
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 

class GMeshLib ; 
class GNodeLib ; 

class GMaker ; 
class GMesh ; 
class GMergedMesh ; 
class GVolume ; 

#include "plog/Severity.h"


/**
GGeoTest
=========

Creates simple test geometries from a commandline specification 
that points to a csgpath directory containing python serialized 
CSG geometry, see for example tboolean-box.

Canonical instance m_geotest resides in OpticksHub and is instanciated
by OpticksHub::loadGeometry only when the --test geometry option is active.  
Which happens after the standard geometry has been loaded.

Rejig
-------

* GGeoTest is now a GGeoBase subclass (just like GGeo and GScene)

* GGeoTest now has its own GGeoLib, to avoid dirty modifyGeometry
  approach which cleared the basis mm

**/


#include "GGeoBase.hh"
#include "GGEO_API_EXPORT.hh"
class GGEO_API GGeoTest : public GGeoBase {
    public:
       static const plog::Severity LEVEL ; 
       static const char* UNIVERSE_PV ; 
       static const char* UNIVERSE_LV ; 
    public:
       // testing utilities used from okg-/OpticksHubTest
       static const char* MakeArgForce(const char* funcname, const char* extra=NULL);
       static std::string MakeArgForce_(const char* funcname, const char* extra);
       static std::string MakeTestConfig_(const char* funcname);
    public:
       GGeoTest(Opticks* ok, GGeoBase* basis=NULL);
       int getErr() const ; 
    private:
       void init();
       void checkPts();

       GMergedMesh* initCreateCSG();
       void addPlaceholderBuffers( GMergedMesh* tmm, unsigned nelem ); 
       void setErr(int err); 
    public:
       // GGeoBase

       GScintillatorLib* getScintillatorLib() const ;
       GSourceLib*       getSourceLib() const ;
       GSurfaceLib*      getSurfaceLib() const ;
       GMaterialLib*     getMaterialLib() const ;
       GMeshLib*         getMeshLib() const ;

       GBndLib*          getBndLib() const ;    
       GGeoLib*          getGeoLib() const ;
       GNodeLib*         getNodeLib() const ;

       const char*       getIdentifier() const ;
       GMergedMesh*      getMergedMesh(unsigned index) const ;
       unsigned          getNumMergedMesh() const ;

    private:
       void autoTestSetup(NCSGList* csglist);
       void relocateSurfaces(GVolume* solid, const char* spec) ;
       void reuseMaterials(NCSGList* csglist);
       void prepareMeshes();
       void adjustContainer();
       void updateWithProxiedSolid();
       void reuseMaterials(const char* spec);
    public:
       void dump(const char* msg="GGeoTest::dump");
    public:
       NGeoTestConfig* getConfig();
       NCSGList*       getCSGList() const ;
       NCSG*           getUniverse() const ;
       NCSG*           findEmitter() const ;
       unsigned        getNumTrees() const ;
       NCSG*           getTree(unsigned index) const ;
    public:
       void anaEvent(OpticksEvent* evt);
       GMergedMesh* combineVolumes( GMergedMesh* mm0);
    private:
       GVolume*     importCSG();
       GMesh*       importMeshViaProxy(NCSG* tree); 
       void         assignBoundaries();


    private:
       Opticks*         m_ok ; 
       bool             m_dbggeotest ;  // --dbggeotest
       const char*      m_config_ ; 
       NGeoTestConfig*  m_config ; 
       unsigned         m_verbosity ;
       OpticksResource* m_resource ; 
       bool             m_dbgbnd ; 
       bool             m_dbganalytic ; 
       NLODConfig*      m_lodconfig ; 
       int              m_lod ; 
       bool             m_input_analytic ; 
       const char*      m_csgpath ; 
       bool             m_test ; 
    private:
       // base geometry and stolen libs 
       GGeoBase*        m_basis ; 
       GMeshLib*        m_basemeshlib ; 
       GGeoLib*         m_basegeolib ; 
   private:
       // local resident libs
       GMaterialLib*    m_mlib ; 
       GSurfaceLib*     m_slib ; 
       GBndLib*         m_bndlib ;  
       GGeoLib*         m_geolib ; 
       GNodeLib*        m_nodelib ; 
       GMeshLib*        m_meshlib ; 
    private:
       // actors
       GMaker*          m_maker ; 
       NCSGList*        m_csglist ; 
       unsigned         m_numtree ; 
       int              m_err ; 

};


