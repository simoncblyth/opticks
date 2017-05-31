#pragma once
#include "NOpenMeshType.hpp"

// NB NOpenMesh closely matches opticks/openmeshrap/MWrap.cc but with less dependencies

#include <map>
struct nnode ; 

#include "NTriSource.hpp"

template <typename T>
struct NPY_API  NOpenMesh : NTriSource
{
    static const char* F_INSIDE_OTHER ; 
    static const char* V_SDF_OTHER ; 
    static const char* V_PARAMETRIC ; 

    enum
    {
        ALL_OUTSIDE_OTHER = 0,
        ALL_INSIDE_OTHER = 7  
    };


    NOpenMesh( const nnode* node, int level, int verbosity, float epsilon=1e-05f ); 

    void init();
    int write(const char* path);
    void dump(const char* msg="NOpenMesh::dump") ;
    void dump_vertices(const char* msg="NOpenMesh::dump_vertices") ;
    void dump_faces(const char* msg="NOpenMesh::dump_faces") ;
    std::string brief();
    std::string desc_inside_other();

    void add_face_(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, typename T::VertexHandle v3, int verbosity=0 );

    typename T::FaceHandle   add_face_(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, int verbosity=0 );

    typename T::VertexHandle add_vertex_unique(typename T::Point pt, bool& added, const float epsilon) ;  
    typename T::VertexHandle find_vertex_exact( typename T::Point pt);
    typename T::VertexHandle find_vertex_closest(typename T::Point pt, float& distance);
    typename T::VertexHandle find_vertex_epsilon(typename T::Point pt, const float epsilon);

    bool is_consistent_face_winding(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2);

    void build_cube();

    void build_parametric();
    void build_parametric_primitive(); 
    void mark_inside_other(const nnode* other);
    void copy_faces(const NOpenMesh<T>* other, int facemask);
    void dump_boundary_faces(const char* msg="NOpenMesh::dump_boundary)_faces", char side='L');

    int  euler_characteristic();


    // NTriSource interface
    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;


    const nnode* node ; 
    int    level ; 
    int    verbosity ;
    float  epsilon ; 

    T              mesh ; 

    NOpenMesh<T>*  leftmesh ; 
    NOpenMesh<T>*  rightmesh ; 

    std::map<int,int> f_inside_other_count ; 


};




