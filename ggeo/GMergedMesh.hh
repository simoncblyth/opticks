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
    static GMergedMesh* create(unsigned ridx, GNode* base, GNode* root, unsigned verbosity);
private:
     // operates in COUNT and MERGE passes, COUNT find out the number of 
     // ridx selected volumes and their vertices to allocate then 
     // MERGE collects them together
     void traverse_r( GNode* node, unsigned int depth, unsigned int pass, unsigned verbosity );

public:
    static GMergedMesh* MakeComposite(std::vector<GMergedMesh*> mms );           // eg for LOD levels 
    static GMergedMesh* MakeLODComposite(GMergedMesh* mm, unsigned levels=3 );   // 2/3 LOD levels 
    static GMergedMesh* CreateBBoxMesh(unsigned index, gbbox& bb );
    static GMergedMesh* CreateQuadMesh(unsigned index, gbbox& bb );
    static bool CheckFacesQty(const GMergedMesh* mm);
public:
    static GMergedMesh* load(Opticks* opticks, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* load(const char* dir, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, const std::vector<GVolume*>& volumes, unsigned verbosity) ;
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, GVolume* volume, unsigned verbosity ) ;
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
    std::string brief() const ;
    void addInstancedBuffers(const std::vector<GNode*>& placements);  // uses GTree statics to create the buffers
   // int  getNumComponents() const ;  <-- this caused some grief, silent override decl without an implementation  
private:
    // NB cannot treat GMergedMesh as a GMesh wrt calling getNumVolumes 
    // explicit naming to avoid subclass confusion
    void countMergedMesh( GMergedMesh* other, bool selected );   
    void countVolume( GVolume*      volume, bool selected, unsigned verbosity  ); 
    void countMesh( const GMesh* mesh ); 
private:
    void mergeVolume( GVolume* volume, bool selected, unsigned verbosity );
    void mergeVolumeIdentity( GVolume* volume, bool selected );
    void mergeVolumeVertices( unsigned nvert, gfloat3* vertices, gfloat3* normals );
    void mergeVolumeFaces( unsigned nface, guint3* faces, unsigned* node_indices, unsigned* boundary_indices, unsigned* sensor_indices );
    void mergeVolumeAnalytic( GParts* pts, GMatrixF* transform, unsigned verbosity );
    void mergeVolumeBBox( gfloat3* vertices, unsigned nvert );
    void mergeVolumeDump( GVolume* volume);
private:
    void mergeMergedMesh( GMergedMesh* other, bool selected, unsigned verbosity );
public:
    float* getModelToWorldPtr(unsigned int index);

    // TODO: below is only usage of GGeo here, move this elsewhere... into GGeo ?
    void reportMeshUsage(GGeo* ggeo, const char* msg="GMergedMesh::reportMeshUsage");
public:
    void dumpVolumes(const char* msg="GMergedMesh::dumpVolumes") const ;
public:
    // used when obtaining relative transforms for flattening sub-trees of repeated geometry
    void   setCurrentBase(GNode* base);
    GNode* getCurrentBase(); 
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
    GNode*       m_cur_base ;  
    std::map<unsigned int, unsigned int> m_mesh_usage ; 

     
};

#include "GGEO_TAIL.hh"

