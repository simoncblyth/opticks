#pragma once


#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

// without this get assert regarding status property on delete_face, see omc-   
#pragma GCC visibility push(default)

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#elif defined(_MSC_VER)

#pragma warning( push )
// OpenMesh/Core/Mesh/AttribKernelT.hh(140): warning C4127: conditional expression is constant
#pragma warning( disable : 4127 )
// OpenMesh/Core/Utils/vector_cast.hh(94): warning C4100: '_dst': unreferenced formal parameter
#pragma warning( disable : 4100 )
// openmesh\core\utils\property.hh(156): warning C4702: unreachable code  
#pragma warning( disable : 4702 )

#endif


#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>



#ifdef __clang__

#pragma clang diagnostic pop
#pragma GCC visibility pop

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning( pop )

#endif



#include "NPY_API_EXPORT.hh"

struct NPY_API NOpenMeshTraits : public OpenMesh::DefaultTraits
{
};

typedef OpenMesh::TriMesh_ArrayKernelT<NOpenMeshTraits>  NOpenMeshType ;


// NB NOpenMesh closely matches opticks/openmeshrap/MWrap.cc
//    but with less dependencies

struct nnode ; 

#include "NTriSource.hpp"

template <typename T>
struct NPY_API  NOpenMesh : NTriSource
{
    int write(const char* path);

    void dump(const char* msg="NOpenMesh::dump") ;
    void dump_vertices(const char* msg="NOpenMesh::dump_vertices") ;
    void dump_faces(const char* msg="NOpenMesh::dump_faces") ;
    std::string brief();

    void add_face_(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, typename T::VertexHandle v3, int verbosity=0 );

    typename T::FaceHandle   add_face_(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, int verbosity=0 );

    typename T::VertexHandle add_vertex_unique(typename T::Point pt, const float epsilon) ;  
    typename T::VertexHandle find_vertex_exact( typename T::Point pt);
    typename T::VertexHandle find_vertex_closest(typename T::Point pt, float& distance);
    typename T::VertexHandle find_vertex_epsilon(typename T::Point pt, const float epsilon);


    bool is_consistent_face_winding(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2);


    void build_cube();
    void build_parametric(const nnode* node, int nu, int nv, int verbosity=0, const float epsilon=1e-5f ); 
    int  euler_characteristic();


    // NTriSource interface
    unsigned get_num_tri() const ;
    unsigned get_num_vert() const ;
    void get_vert( unsigned i, glm::vec3& v ) const ;
    void get_tri( unsigned i, glm::uvec3& t ) const ;
    void get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const ;


    T    mesh ; 
};




