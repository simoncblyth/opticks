#pragma once
#include "NOpenMeshType.hpp"

// NB NOpenMesh closely matches opticks/openmeshrap/MWrap.cc but with less dependencies

struct nnode ; 
struct nuv ; 

#include "NTriSource.hpp"

#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshFind.hpp"
#include "NOpenMeshBuild.hpp"
#include "NOpenMeshSubdiv.hpp"


template <typename T>
struct NPY_API  NOpenMesh : NTriSource
{
    static NOpenMesh<T>* hexpatch(int level, int verbosity, int ctrl );
    static NOpenMesh<T>* cube(int level, int verbosity, int ctrl );
    static NOpenMesh<T>* tetrahedron(int level, int verbosity, int ctrl  ) ;

    NOpenMesh( const nnode* node, int level, int verbosity, int ctrl=0, float epsilon=1e-05f ); 

    void init();
    void check();
    int write(const char* path);
    void dump(const char* msg="NOpenMesh::dump") ;
    std::string brief();

    void build_parametric();


    void dump_border_faces(const char* msg="NOpenMesh::dump_border_faces", char side='L');


    void subdiv_test() ;
    void subdiv_interior_test() ;
    void subdivide_border_faces(const nnode* other, unsigned nsubdiv );


    // NTriSource interface
    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;


    T                  mesh ; 
    NOpenMeshProp<T>   prop ; 
    NOpenMeshDesc<T>   desc ; 
    NOpenMeshFind<T>   find ; 
    NOpenMeshBuild<T>  build ; 
    NOpenMeshSubdiv<T> subdiv ; 

    const nnode* node ; 
    int    level ; 
    int    verbosity ;
    int    ctrl ;
    float  epsilon ; 
    unsigned nsubdiv ; 

    NOpenMesh<T>*  leftmesh ; 
    NOpenMesh<T>*  rightmesh ; 

};




