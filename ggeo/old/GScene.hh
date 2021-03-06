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

#include <string>
#include <map>

class SLog ; 

class Opticks ; 
class OpticksQuery ; 
class OpticksEvent ; 

class GNodeLib ; 
class GMeshLib ; 
class GItemList ; 
class GGeo ; 
class GVolume ; 
class GNode ; 

class GSurfaceLib ; 
class GMaterialLib ; 
class GBndLib ; 
class GPmtLib ; 
class GGeoLib ; 
class GMeshLib ; 

class GMesh ; 
class GMergedMesh ; 
class GItemIndex ; 
class GColorizer ; 

class NCSG ; 
class NSensorList ; 
class NScene ; 
struct NSceneConfig ; 
struct nd ; 
struct guint4 ; 

template<class T> class NPY ;


#include "NSceneConfig.hpp"  // for enum
#include "GGeoBase.hh"

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/*
GScene
========

* GGeoBase : protocal base class ensuring common access from GGeo and GScene 


Canonical m_gscene instance, resident in m_ggeo,
is instanciated by GGeo::loadFromGLTF.
GMergedMesh are currently created via GGeo and
managed in its GGeoLib.

Note that GScene does very little postcache, it does its work
precache putting together the GParts and associating them
with the GMergedMesh which get persisted by the GGeoLib
into the legacy GMergedMeshAnalytic.


Legacy Workflow Reminder
--------------------------

GScene is only used in legacy geometry workflow. It is intended 
for legacy workflow and GScene to be eliminated, 
but thats going to require major surgery : so for now have to live with it.

Legacy workflow has separate triangulated and analytic geometry routes, whereas
the direct workflow does these together. For more of a reminder 

* notes/issues/plan-removal-of-legacy-geometry-workflow-packages-and-externals.rst


Two very distict modes : Loaded/Created
------------------------------------------

::

      
     116     m_geolib(loaded ? GGeoLib::Load(m_ok, m_analytic, m_tri_bndlib ) : new GGeoLib(m_ok, m_analytic, m_tri_bndlib)),
     117     m_nodelib(loaded ? GNodeLib::Load(m_ok, m_analytic, m_testgeo ) : new GNodeLib(m_ok, m_analytic, m_testgeo )),
     118     m_meshlib(loaded ? GMeshLib::Load(m_ok, m_analytic)  : new GMeshLib(m_ok, m_analytic)),
      

GScene.hh only used from GGeo, the actions of GScene
creating the analytic GMergedMesh are felt via 
the normal GGeoLib route. oxrap/OScene/OGeo 
(especially OGeo::makeAnalyticGeometry)
which converts the GGeo accessed GMergedMesh into OptiX form. 

Fully analytic glTF based replacement for the 
mainly triangulated GInstancer.

Note the only public method is the ctor, this 
gets invoked from GGeo::loadGeometry/GGeo::loadFromGLTF
when the "--gltf 1" commandline option is used with a 
value greater than 0.

Using "--gltf 4", signals an early exit following GScene 
instanciation in GGeo::loadFromGLTF.


* generally follows the same pattern as GInstancer 
  labelling the GNode tree with a ridx repeat index... 
  which us used within  GGeo/GGeoLib/GMergedMesh mesh merging
  (which also merges analytic solids)

* a global mm0 is needed for setting domains... when 
  operating purely instanced need to construct some 
  placeholder bbox so satisfy the global mesh 0 that 
  lots of things require



void GScene::initFromGLTF()
------------------------------



GVolume* GScene::createVolumeTree(NScene* scene) 
-------------------------------------------------

* recursively convert nd tree into GVolume(GNode) tree
* recall GVolume(GNode) was formerly misnamed as "GSolid"


GVolume* GScene::createVolume(nd* n, unsigned depth, bool& recursive_select  )
------------------------------------------------------------------------------

* nexus : GMesh, GVolume, GParts, NCSG, boundary spec  





*/


#include "plog/Severity.h"


class GGEO_API GScene : public GGeoBase
{
   //    friend class GGeo ;
        static const plog::Severity LEVEL ;   
    public:
        static bool HasCache( Opticks* ok ); 
        static GScene* Create(Opticks* ok, GGeo* ggeo) ;
        static GScene* Load(Opticks* ok, GGeo* ggeo) ;
        GScene(Opticks* ok, GGeo* ggeo, bool loaded);

        const GVolume* getVolume(unsigned nidx);
        void dumpNode(unsigned nidx);
        void anaEvent(OpticksEvent* evt);
        void save() const ; 

    public:
        // GGeoBase interface
        GScintillatorLib* getScintillatorLib() const ; 
        GSourceLib*       getSourceLib() const ; 
        GSurfaceLib*      getSurfaceLib() const ; 
        GMaterialLib*     getMaterialLib() const ; 

        GBndLib*          getBndLib() const ; 
        //GPmtLib*          getPmtLib() const ; 
        GGeoLib*          getGeoLib() const ; 
        GNodeLib*         getNodeLib() const ;
        GMeshLib*         getMeshLib() const ;

        const char*       getIdentifier() const ;
        GMergedMesh*      getMergedMesh(unsigned ridx) const ;

        // GGeoBase interface END
    private:
        void initFromGLTF();
        void prepareVertexColors();
    private:
        void dumpTriInfo() const ; 
        void compareTrees() const ;
        void modifyGeometry();
        void importMeshes(NScene* scene);
        void dumpMeshes();
        void compareMeshes();
        void compareMeshes_GMeshBB();
        const GMesh* getMesh(unsigned mesh_idx);
        unsigned getNumMeshes();

        NCSG* getCSG(unsigned mesh_idx);
        NCSG* findCSG(const char* soname, bool startswith) const ;
        nbbox getBBox(const char* soname, NSceneConfigBBoxType bbty) const ; 


        unsigned findTriMeshIndex(const char* soname) const ;

        // from triangulated branch mm0
        guint4 getNodeInfo(unsigned idx) const ;
        guint4 getIdentity(unsigned idx) const ;
    private:
        GVolume* createVolumeTree(NScene* scene);
        GVolume* createVolumeTree_r(nd* n, GVolume* parent, unsigned depth, bool recursive_select );
        GVolume* createVolume(nd* n, unsigned depth, bool& recursive_select );
        void transferIdentity( GVolume* node, const nd* n);
        void transferMetadata( GVolume* node, const NCSG* csg, const nd* n, unsigned depth, bool& recursive_select );
        std::string lookupBoundarySpec( const GVolume* node, const nd* n) const ;
        void addNode(GVolume* node, nd* n);
    private:
        // compare tree calculated and persisted transforms
        void           deltacheck_r( GNode* node, unsigned int depth );
    private:
        void         checkMergedMeshes();
        void         makeMergedMeshAndInstancedBuffers() ; 

    private:
        GVolume*       getNode(unsigned node_idx);
    private:
        SLog*         m_log ;   
        Opticks*      m_ok ; 
        OpticksQuery* m_query ; 
        
        GGeo*    m_ggeo ; 

        NSensorList*  m_sensor_list ; 

        GGeoLib*      m_tri_geolib ; 
        GMergedMesh*  m_tri_mm0 ; 
        GNodeLib*     m_tri_nodelib ; 
        GBndLib*      m_tri_bndlib ; 
        GMeshLib*     m_tri_meshlib ; 
        GItemIndex*   m_tri_meshindex ; 

     
        bool     m_analytic ; 
        bool     m_testgeo ; 
        bool     m_loaded ; 
        bool     m_honour_selection ;

        int      m_gltf ; 
        NSceneConfig*  m_scene_config ; 
        NScene*        m_scene ; 
        int            m_num_nd ; 
        unsigned       m_targetnode ; 

        GGeoLib*      m_geolib ; 
        GNodeLib*     m_nodelib ; 
        GMeshLib*     m_meshlib ; 

        GColorizer*   m_colorizer ; 

        unsigned     m_verbosity ; 
        GVolume*     m_root ; 
        unsigned     m_selected_count ; 

        std::map<unsigned, GMesh*>   m_meshes ; 
        std::map<unsigned, GVolume*>  m_nodes ;  
        std::map<unsigned, unsigned> m_rel2abs_mesh ; 
        std::map<unsigned, unsigned> m_abs2rel_mesh ; 

};




#include "GGEO_TAIL.hh"


