#pragma once

#include <vector>

#include "NOpenMeshCfg.hpp"
#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"
#include "NOpenMeshBoundary.hpp"

template <typename T> struct NOpenMeshProp ;

template <typename T>
struct NPY_API  NOpenMeshFind
{
    typedef typename T::FaceHandle FH ; 
    typedef typename T::Point      P ; 
    typedef typename T::VertexHandle         VH ; 
    typedef typename T::ConstFaceVertexIter  FVI ; 
    typedef typename T::ConstFaceFaceIter   CFFI ;  // cff_iter(fh)

    NOpenMeshFind( T& mesh, 
                   const NOpenMeshCfg&      cfg,
                   const NOpenMeshProp<T>& prop, 
                   int verbosity );

    int find_boundary_loops() ;

    void find_faces(std::vector<FH>& faces, NOpenMeshFindType find, int param) const ;
    void sort_faces(           std::vector<FH>& faces) const ;
    void sort_faces_contiguous(std::vector<FH>& faces) const ;
    bool are_contiguous(const FH a, const FH b) const ;
    void dump_contiguity( const std::vector<FH>& faces ) const ;


    bool is_numboundary_face(const FH fh, int numboundary) const ;
    bool is_regular_face(const FH fh, int valence ) const ;
    bool is_interior_face(const FH fh, int margin ) const ;

    typename T::VertexHandle find_vertex_exact( P pt) const ;
    typename T::VertexHandle find_vertex_closest(P pt, float& distance) const ;
    typename T::VertexHandle find_vertex_epsilon(P pt, const float epsilon) const ;


    T&                      mesh  ;
    const NOpenMeshCfg&      cfg ; 
    const NOpenMeshProp<T>& prop ;
    int                     verbosity ; 

    std::vector<NOpenMeshBoundary<T>> loops ; 


};
 


