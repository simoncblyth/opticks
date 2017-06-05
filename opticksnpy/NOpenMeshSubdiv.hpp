#pragma once

#include <string>
#include "NOpenMeshType.hpp"

namespace OpenMesh {
namespace Subdivider {
namespace Adaptive   {
    template <typename T> class CompositeT ;
}
}
}

template <typename T> struct NOpenMeshProp ;
template <typename T> struct NOpenMeshDesc ;
template <typename T> struct NOpenMeshFind ;
template <typename T> struct NOpenMeshBuild ;


template <typename T>
struct NPY_API  NOpenMeshSubdiv
{
    typedef typename OpenMesh::Subdivider::Adaptive::CompositeT<T> subdivider_t ;  

    NOpenMeshSubdiv( T& mesh, 
                     NOpenMeshProp<T>& prop, 
                     const NOpenMeshDesc<T>& desc, 
                     const NOpenMeshFind<T>& find, 
                     NOpenMeshBuild<T>& build, 
                     int verbosity,
                     float epsilon
                    );

    void init();
    void init_subdivider();
    std::string brief();

    void refine(typename T::FaceHandle fh); 


    typename T::VertexHandle centroid_split_face(typename T::FaceHandle fh );

    void sqrt3_split_r( typename T::FaceHandle fh, const nnode* other );
    void create_soup(typename T::FaceHandle fh, const nnode* other );



    T& mesh  ;
    NOpenMeshProp<T>& prop ;
    const NOpenMeshDesc<T>& desc ;
    const NOpenMeshFind<T>& find ;
    NOpenMeshBuild<T>& build ;
    int verbosity ; 
    float epsilon ; 

    subdivider_t* subdivider ; 

};
 



