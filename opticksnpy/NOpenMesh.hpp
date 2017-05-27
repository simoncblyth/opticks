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

template <typename T>
struct NPY_API  NOpenMesh
{
    int write(const char* path);

    void dump(const char* msg="NOpenMesh::dump") ;
    std::string brief();
    void dump_vertices(const char* msg="NOpenMesh::dump_vertices") ;
    void dump_faces(const char* msg="NOpenMesh::dump_faces") ;

    void add_face(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, typename T::VertexHandle v3 );
    typename T::VertexHandle add_vertex_unique(typename T::Point pt) ;  
    typename T::VertexHandle find_vertex(typename T::Point pt);

    void build_cube();
    void build_parametric(const nnode* node, int usteps, int vsteps); 


    T mesh ; 
};




