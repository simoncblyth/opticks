#pragma once

#include <vector>

#include "NOpenMeshType.hpp"
#include "NOpenMeshBoundary.hpp"

template <typename T> struct NOpenMeshProp ;

typedef enum {
   FIND_ALL_FACE,
   FIND_REGULAR_FACE,
   FIND_INTERIOR_FACE
} select_t ; 


template <typename T>
struct NPY_API  NOpenMeshFind
{
    NOpenMeshFind( T& mesh, const NOpenMeshProp<T>& prop );

    int find_boundary_loops() ;

    void find_faces(std::vector<typename T::FaceHandle>& faces, select_t select, unsigned param);
    bool is_regular_face(const typename T::FaceHandle& fh, unsigned valence );
    bool is_interior_face(const typename T::FaceHandle& fh, unsigned margin );

    typename T::VertexHandle find_vertex_exact( typename T::Point pt) const ;
    typename T::VertexHandle find_vertex_closest(typename T::Point pt, float& distance) const ;
    typename T::VertexHandle find_vertex_epsilon(typename T::Point pt, const float epsilon) const ;



    T& mesh  ;
    const NOpenMeshProp<T>& prop ;

    std::vector<NOpenMeshBoundary<T>> loops ; 


};
 


