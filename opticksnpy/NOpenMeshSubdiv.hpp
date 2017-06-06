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
    typedef typename T::Point             P ; 
    typedef typename T::EdgeHandle       EH ; 
    typedef typename T::VertexHandle     VH ; 
    typedef typename T::HalfedgeHandle  HEH ; 
    typedef typename T::FaceHandle      FH ; 
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

    void refine(FH fh); 

    void sqrt3_refine( std::vector<FH>& faces );
    void sqrt3_split_r( FH fh, int depth );
    void sqrt3_flip_edge(HEH heh);


    typename T::FaceHandle next_opposite_face(HEH heh);
    std::string desc_face(const FH fh, const char* label);

    void create_soup(FH fh, const nnode* other );



    T&                       mesh ;
    NOpenMeshProp<T>&        prop ;
    const NOpenMeshDesc<T>&  desc ;
    const NOpenMeshFind<T>&  find ;
    NOpenMeshBuild<T>&       build ;
    int                      verbosity ; 
    float                    epsilon ; 

    subdivider_t*            subdivider ; 

};
 



