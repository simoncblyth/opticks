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







#ifdef OLD_PARAMETERS
class X_BParameters ; 
#else
class NMeta ; 
#endif


template <typename T> struct NOpenMeshZipper ; 

/*
NOpenMesh
===========

Canonically used from 

   NCSG::LoadTree
   NCSG::polygonize
   NPolygonizer::hybridMesher
   NHybridMesher::make_mesh


*/

template <typename T>
struct NPY_API  NOpenMesh : NTriSource
{
    typedef NOpenMesh<T>    MESH ; 
    typedef typename T::Point           P ; 
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::FaceHandle     FH ; 


#ifdef OLD_PARAMETERS
    static NOpenMesh<T>* Make( const nnode* node, const X_BParameters* meta, const char* treedir );
#else
    static NOpenMesh<T>* Make( const nnode* node, const NMeta* meta, const char* treedir );
#endif

    NOpenMesh<T>* make_submesh(   const nnode* subnode);
    NOpenMesh<T>* make_selection( NOpenMeshPropType select );

    NOpenMesh( const nnode* node, const NOpenMeshCfg* cfg, bool partial ); 

    void build_csg();
    void init();
    void check();

    int write(const char* path) const ;
    void save(const char* name="mesh.off") const ;

    void dump(const char* msg="NOpenMesh::dump") const ;
    std::string brief() const ;  
    std::string summary(const char* msg="NOpenMesh::summary") const ; 


    void combine_hybrid();
    void combine_csgbsp();


    void dump_border_faces(const char* msg="NOpenMesh::dump_border_faces", char side='L');

    void subdiv_test() ;



    // from find
    unsigned              get_num_boundary_loops() const ;
    const NOpenMeshBoundary<T>& get_boundary_loop(unsigned i) const ;


    // NTriSource interface
    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_normal( unsigned i, glm::vec3& n ) const ;
    void get_uv(  unsigned i, glm::vec3& uv ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;


    T                   mesh ; 

    const nnode*        node ; 
    const NOpenMeshCfg* cfg ; 
    const bool          partial ; 
    const int           verbosity ;

    NOpenMeshProp<T>   prop ; 
    NOpenMeshDesc<T>   desc ; 
    NOpenMeshFind<T>   find ; 
    NOpenMeshBuild<T>  build ; 
    NOpenMeshSubdiv<T> subdiv ; 



    MESH*  leftmesh ; 
    MESH*  rightmesh ; 

    MESH*  lfrontier ; 
    MESH*  rfrontier ; 

    NOpenMeshZipper<T>* zipper ; 


};




