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
#include <vector>
#include <string>
#include "plog/Severity.h"

template <typename T> class NPY ;

class Opticks ; 

class GGeo ; 
class GNode ;
class GVolume ; 
class GMergedMesh ; 
class GPts ; 

#include "GMesh.hh"
#include "GVector.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**
GMergedMesh
=============

* creation of composite meshes from multiple GMesh 
* general usage should target GMesh  
* THAT MEANS : DO NOT ADD METHODS HERE THAT CAN LIVE IN GMesh

Usage of GMergedMesh
-----------------------

GGeoLib::makeMergedMesh
    canonical driver 


**/

class GGEO_API GMergedMesh : public GMesh {

    friend class GGeoLib ;         // for setParts hookup on loading 
    friend class GGeoTest ;         // for setParts analytic PMT kludge
    friend class OpticksGeometry ;  // for setParts analytic PMT kludge
public:
    enum { PASS_COUNT, PASS_MERGE } ;
public:
    static const plog::Severity LEVEL ; 
    static std::string Desc(const GMergedMesh* mm);
    static GMergedMesh* Create(unsigned ridx, const GNode* base, const GNode* root );
private:
     // operates in COUNT and MERGE passes, COUNT find out the number of 
     // ridx selected volumes and their vertices to allocate then 
     // MERGE collects them together
     void traverse_r( const GNode* node, unsigned int depth, unsigned int pass );
     void postcreate(); 
public:
    static GMergedMesh* MakeComposite(std::vector<GMergedMesh*> mms );           // eg for LOD levels 
    static GMergedMesh* MakeLODComposite(GMergedMesh* mm, unsigned levels=3 );   // 2/3 LOD levels 
    static GMergedMesh* CreateBBoxMesh(unsigned index, gbbox& bb );
    static GMergedMesh* CreateQuadMesh(unsigned index, gbbox& bb );
    static bool CheckFacesQty(const GMergedMesh* mm);
public:
    static GMergedMesh* Load(Opticks* opticks, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* Load(const char* dir, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* Combine(unsigned int index, GMergedMesh* mm, const std::vector<GVolume*>& volumes ) ;
    static GMergedMesh* Combine(unsigned int index, GMergedMesh* mm, GVolume* volume ) ;
public:
    GMergedMesh(unsigned index) ;
    GMergedMesh(                // expedient pass-thru to GMesh ctor
             unsigned index, 
             gfloat3* vertices, 
             unsigned num_vertices, 
             guint3*  faces, 
             unsigned num_faces, 
             gfloat3* normals, 
             gfloat2* texcoords
         );
public:
    char getCurrentGeoCode() const ;
    std::string brief() const ;
    void addInstancedBuffers(const std::vector<const GNode*>& placements);  // uses GTree statics to create the buffers
   // int  getNumComponents() const ;  <-- this caused some grief, silent override decl without an implementation  
private:
    // NB cannot treat GMergedMesh as a GMesh wrt calling getNumVolumes 
    // explicit naming to avoid subclass confusion
    void init(); 
    void countMergedMesh( GMergedMesh* other, bool selected );   
    void countVolume( const GVolume*      volume, bool selected ); 
    void countMesh( const GMesh* mesh ); 
private:
    void mergeVolume( const GVolume* volume, bool selected );
    void mergeVolumeIdentity( const GVolume* volume, bool selected );
    void mergeVolumeVertices( unsigned nvert, gfloat3* vertices, gfloat3* normals );
    void mergeVolumeFaces( unsigned nface, guint3* faces, unsigned* node_indices, unsigned* boundary_indices, unsigned* sensor_indices );

#ifdef GPARTS_HOT
    void mergeVolumeAnalytic( GParts* pts, GMatrixF* transform );
#endif

    void mergeVolumeAnalytic( GPt*    pt,  GMatrixF* transform );
    void mergeVolumeTransform( GMatrixF* transform ); 
    void mergeVolumeBBox( gfloat3* vertices, unsigned nvert );
    void mergeVolumeDump( const GVolume* volume);

private:
    void mergeMergedMesh( GMergedMesh* other, bool selected );
public:
    float* getModelToWorldPtr(unsigned int index);

    // TODO: below is only usage of GGeo here, move this elsewhere... into GGeo ?
    void reportMeshUsage(GGeo* ggeo, const char* msg="GMergedMesh::reportMeshUsage");
public:
    void dumpVolumesSelected(const char* msg="GMergedMesh::dumpVolumesSelected") const ;
    void dumpVolumes(const char* msg="GMergedMesh::dumpVolumes") const ;
    void dumpVolumesFaces(const char* msg="GMergedMesh::dumpVolumesFaces") const  ;  // migrated from OGeo
    void dumpTransforms( const char* msg="GMergedMesh::dumpTransforms") const ; // migrated from OGeo

    void setPts(GPts* pts); 
    GPts* getPts() const ; 

public:
    // used when obtaining relative transforms for flattening sub-trees of repeated geometry
    void   setCurrentBase(const GNode* base);
    const GNode* getCurrentBase(); 

    bool   isGlobal(); 
    bool   isInstanced(); 
public:
    // geocodes used to communicate between ggv- oglrap- optixrap-
    bool   isSkip() const ; 
    bool   isAnalytic() const ; 
    bool   isTriangulated() const ; 
private:
    // transients that do not need persisting, persistables are down in GMesh
    unsigned     m_cur_vertices ;
    unsigned     m_cur_faces ;
    unsigned     m_cur_volume ;
    unsigned     m_cur_mergedmesh ; // for composite mergedmesh recording 
    unsigned     m_num_csgskip ; 
    const GNode* m_cur_base ;  
    std::map<unsigned int, unsigned int> m_mesh_usage ; 

    GPts*        m_pts ; 
    Opticks*     m_ok ; 

     
};

#include "GGEO_TAIL.hh"

