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

struct NOpenMeshCfg ;
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
    typedef typename T::VertexOHalfedgeIter   VOHI ;

    //typedef typename T::ConstVertexFaceIter CVFI ;  // cvf_iter(vh)
    //typedef typename T::ConstFaceFaceIter   CFFI ;  // cff_iter(fh)


    typedef typename OpenMesh::Subdivider::Adaptive::CompositeT<T> subdivider_t ;  

    NOpenMeshSubdiv( T& mesh, 
                     const NOpenMeshCfg& cfg,
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
    void sqrt3_refine( NOpenMeshFindType select, int param  );
    void sqrt3_refine_phased( const std::vector<FH>& target );
    void sqrt3_refine_contiguous( std::vector<FH>& target );


    void sqrt3_split_r( FH fh, int depth );

    void sqrt3_centroid_split_face(FH fh, std::vector<VH>& centroid_vertices);
    void sqrt3_flip_adjacent_edges(const VH cvh, int maxflip);
    void sqrt3_flip_edge(HEH heh);


    std::string desc_face(const FH fh, const char* label);
    std::string desc_vertex(const VH vh, const char* label);
    std::string desc_edge(const EH eh, const char* label);


    typename T::FaceHandle next_opposite_face(HEH heh);

    void create_soup(FH fh, const nnode* other );



    T&                       mesh ;
    const NOpenMeshCfg&      cfg ; 
    NOpenMeshProp<T>&        prop ;
    const NOpenMeshDesc<T>&  desc ;
    const NOpenMeshFind<T>&  find ;
    NOpenMeshBuild<T>&       build ;
    int                      verbosity ; 
    float                    epsilon ; 

    subdivider_t*            subdivider ; 

};
 



