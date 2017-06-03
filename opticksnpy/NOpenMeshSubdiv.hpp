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


template <typename T>
struct NPY_API  NOpenMeshSubdiv
{
    typedef typename OpenMesh::Subdivider::Adaptive::CompositeT<T> subdivider_t ;  

    NOpenMeshSubdiv( T& mesh );
    void init();
    std::string desc();

    void refine(typename T::FaceHandle fh); 


    T& mesh  ;

    subdivider_t* subdivider ; 

};
 



