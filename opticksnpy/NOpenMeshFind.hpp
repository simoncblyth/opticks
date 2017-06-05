#pragma once

#include <vector>

#include "NOpenMeshType.hpp"
#include "NOpenMeshBoundary.hpp"

template <typename T> struct NOpenMeshProp ;

typedef enum {
   FIND_ALL_FACE,
   FIND_IDENTITY_FACE,
   FIND_NONBOUNDARY_FACE,
   FIND_BOUNDARY_FACE,
   FIND_REGULAR_FACE,
   FIND_INTERIOR_FACE
} select_t ; 


template <typename T>
struct NPY_API  NOpenMeshFind
{
    typedef typename T::FaceHandle FH ; 
    typedef typename T::Point      P ; 

    NOpenMeshFind( T& mesh, const NOpenMeshProp<T>& prop );

    int find_boundary_loops() ;

    void find_faces(std::vector<FH>& faces, select_t select, int param);

    bool is_numboundary_face(const FH fh, int numboundary);
    bool is_regular_face(const FH fh, int valence );
    bool is_interior_face(const FH fh, int margin );

    typename T::VertexHandle find_vertex_exact( P pt) const ;
    typename T::VertexHandle find_vertex_closest(P pt, float& distance) const ;
    typename T::VertexHandle find_vertex_epsilon(P pt, const float epsilon) const ;


    T& mesh  ;
    const NOpenMeshProp<T>& prop ;

    std::vector<NOpenMeshBoundary<T>> loops ; 


};
 


