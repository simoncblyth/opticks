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
                     const NOpenMeshProp<T>& prop, 
                     const NOpenMeshDesc<T>& desc, 
                     const NOpenMeshFind<T>& find, 
                     NOpenMeshBuild<T>& build );

    void init();
    void init_subdivider();
    std::string brief();

    void refine(typename T::FaceHandle fh); 

    void manual_subdivide_face(              typename T::FaceHandle fh, const nnode* other, int verbosity, float epsilon );
    void manual_subdivide_face_creating_soup(typename T::FaceHandle fh, const nnode* other, int verbosity, float epsilon );





    T& mesh  ;
    const NOpenMeshProp<T>& prop ;
    const NOpenMeshDesc<T>& desc ;
    const NOpenMeshFind<T>& find ;
    NOpenMeshBuild<T>& build ;

    subdivider_t* subdivider ; 

};
 



