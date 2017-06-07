#pragma once
#include "NOpenMeshType.hpp"

// NB NOpenMesh closely matches opticks/openmeshrap/MWrap.cc but with less dependencies

struct nnode ; 
struct nuv ; 

#include "NTriSource.hpp"

#include "NOpenMeshEnum.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshFind.hpp"
#include "NOpenMeshBuild.hpp"
#include "NOpenMeshSubdiv.hpp"
#include "NOpenMeshCfg.hpp"


typedef enum {
   COMBINE_HYBRID = 0x1 << 0,
   COMBINE_CSGBSP = 0x1 << 1
} NOpenMeshMode_t ; 


template <typename T>
struct NPY_API  NOpenMesh : NTriSource
{
    static const char* COMBINE_HYBRID_ ; 
    static const char* COMBINE_CSGBSP_ ; 
    static const char* MeshModeString(NOpenMeshMode_t meshmode);

    typedef NOpenMesh<T>    MESH ; 
    typedef typename T::Point           P ; 
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::FaceHandle     FH ; 

    static NOpenMesh<T>* BuildTest(int level, int verbosity, int ctrl, const char* polycfg);

    NOpenMesh<T>* spawn(const nnode* subnode);
    NOpenMesh( const nnode* node, int level, int verbosity, int ctrl=0, const char* polycfg=NULL, NOpenMeshMode_t meshmode=COMBINE_HYBRID, float epsilon=1e-05f ); 

    void build_csg();
    void init();
    void check();

    int write(const char* path);
    void dump(const char* msg="NOpenMesh::dump") ;
    std::string brief();

    void combine_hybrid();
    void combine_csgbsp();


    void dump_border_faces(const char* msg="NOpenMesh::dump_border_faces", char side='L');

    void subdiv_test() ;


    // NTriSource interface
    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_normal( unsigned i, glm::vec3& n ) const ;
    void get_uv(  unsigned i, glm::vec3& uv ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;


    T                  mesh ; 
    NOpenMeshCfg       cfg ; 
    NOpenMeshProp<T>   prop ; 
    NOpenMeshDesc<T>   desc ; 
    NOpenMeshFind<T>   find ; 
    NOpenMeshBuild<T>  build ; 
    NOpenMeshSubdiv<T> subdiv ; 

    const nnode*    node ; 
    int             level ; 
    int             verbosity ;
    int             ctrl ;
    NOpenMeshMode_t meshmode ; 
    float           epsilon ; 


    MESH*  leftmesh ; 
    MESH*  rightmesh ; 

};




