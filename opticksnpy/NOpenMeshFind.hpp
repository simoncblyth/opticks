#pragma once

#include <vector>

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

    NOpenMeshFind( T& mesh, const NOpenMeshProp<T>& prop, int verbosity );

    int find_boundary_loops() ;

    void find_faces(std::vector<FH>& faces, NOpenMeshFindType find, int param);

    bool is_numboundary_face(const FH fh, int numboundary);
    bool is_regular_face(const FH fh, int valence );
    bool is_interior_face(const FH fh, int margin );

    typename T::VertexHandle find_vertex_exact( P pt) const ;
    typename T::VertexHandle find_vertex_closest(P pt, float& distance) const ;
    typename T::VertexHandle find_vertex_epsilon(P pt, const float epsilon) const ;


    T&                      mesh  ;
    const NOpenMeshProp<T>& prop ;
    int                     verbosity ; 

    std::vector<NOpenMeshBoundary<T>> loops ; 


};
 


