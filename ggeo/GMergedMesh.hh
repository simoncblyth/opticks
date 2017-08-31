#pragma once

// *GMergedMesh* 
//     is just relevant for GMesh creation from multiple GMesh general usage should target GMesh  
//
//     THAT MEANS : DO NOT ADD METHODS HERE THAT CAN LIVE IN GMesh

#include <map>
#include <vector>
#include <string>


template <typename T> class NPY ;

class Opticks ; 

class GGeo ; 
class GNode ;
class GSolid ; 
class GParts ; 
class GMergedMesh ; 

#include "GMesh.hh"
#include "GVector.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"


class GGEO_API GMergedMesh : public GMesh {
    friend class GGeoLib ;         // for setParts hookup on loading 
    friend class GGeoTest ;         // for setParts analytic PMT kludge
    friend class OpticksGeometry ;  // for setParts analytic PMT kludge
public:
    enum { PASS_COUNT, PASS_MERGE } ;
public:
    static GMergedMesh* create(unsigned ridx, GNode* base, GNode* root, unsigned verbosity);
private:
     // operates in COUNT and MERGE passes, COUNT find out the number of 
     // ridx selected solids and their vertices to allocate then 
     // MERGE collects them together
     void traverse_r( GNode* node, unsigned int depth, unsigned int pass, unsigned verbosity );

public:
    static GMergedMesh* MakeComposite(std::vector<GMergedMesh*> mms ); // eg for LOD levels 
public:
    static GMergedMesh* load(Opticks* opticks, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* load(const char* dir, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, const std::vector<GSolid*>& solids, unsigned verbosity) ;
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, GSolid* solid, unsigned verbosity ) ;
public:
    GMergedMesh(unsigned int index) ;
    GParts* getParts();
    std::string brief() const ;
    void addInstancedBuffers(const std::vector<GNode*>& placements);
    int  getNumComponents() const ;
private:
    void setParts(GParts* pts); 
private:
    // NB cannot treat GMergedMesh as a GMesh wrt calling getNumSolids 
    // explicit naming to avoid subclass confusion
    void countMergedMesh( GMergedMesh* other, bool selected );   
    void countSolid( GSolid*      solid, bool selected, unsigned verbosity  ); 
    void countMesh( const GMesh* mesh ); 
private:
    void mergeSolid( GSolid* solid, bool selected, unsigned verbosity );
    void mergeSolidIdentity( GSolid* solid, bool selected );
    void mergeSolidVertices( unsigned nvert, gfloat3* vertices, gfloat3* normals );
    void mergeSolidFaces( unsigned nface, guint3* faces, unsigned* node_indices, unsigned* boundary_indices, unsigned* sensor_indices );
    void mergeSolidAnalytic( GParts* pts, GMatrixF* transform, unsigned verbosity );
    void mergeSolidBBox( gfloat3* vertices, unsigned nvert );
    void mergeSolidDump( GSolid* solid);
private:
    void mergeMergedMesh( GMergedMesh* other, bool selected );
public:
    float* getModelToWorldPtr(unsigned int index);

    // TODO: below is only usage of GGeo here, move this elsewhere... into GGeo ?
    void reportMeshUsage(GGeo* ggeo, const char* msg="GMergedMesh::reportMeshUsage");
public:
    void dumpSolids(const char* msg="GMergedMesh::dumpSolids") const ;
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
    unsigned     m_cur_solid ;
    unsigned     m_cur_mergedmesh ; // for composite mergedmesh recording 
    unsigned     m_num_csgskip ; 
    GNode*       m_cur_base ;  
    GParts*      m_parts ; 
    std::map<unsigned int, unsigned int> m_mesh_usage ; 

     
};

#include "GGEO_TAIL.hh"

