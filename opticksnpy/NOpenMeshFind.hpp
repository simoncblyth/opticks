#pragma once

#include <vector>

#include "NOpenMeshCfg.hpp"
#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"
#include "NOpenMeshBoundary.hpp"

struct nnode ; 
template <typename T> struct NOpenMeshProp ;

template <typename T>
struct NPY_API  NOpenMeshFind
{
    typedef typename T::FaceHandle FH ; 
    typedef typename T::Point      P ; 
    typedef typename T::VertexHandle         VH ; 
    typedef typename T::ConstFaceVertexIter  CFVI ; 
    typedef typename T::ConstFaceFaceIter    CFFI ;  // cff_iter(fh)

    NOpenMeshFind( T& mesh, 
                   const NOpenMeshCfg&      cfg,
                   NOpenMeshProp<T>& prop, 
                   int verbosity, 
                   const nnode* node);



    void                  dump_boundary_loops(const char* msg="NOpenMeshFind::dump_boundary_loops", bool detail=false) ;
    int                   find_boundary_loops() ;
    unsigned              get_num_boundary_loops();
    NOpenMeshBoundary<T>& get_boundary_loop(unsigned i);




    bool                    is_selected(const FH fh, NOpenMeshFindType sel, int param) const ;
   

    std::string desc_face(const FH fh) const ;
    std::string desc_face_i(const FH fh) const ;
    std::string desc_face_v(const FH fh) const ;

    typename T::FaceHandle first_face(const std::vector<FH>& faces, NOpenMeshFindType sel, int param) const ;
    typename T::FaceHandle first_face(                              NOpenMeshFindType sel, int param) const ;
     // NB check mesh.is_valid_handle to see if a face was found

    void find_faces(std::vector<FH>& faces, NOpenMeshFindType find, int param) const ;
    void sort_faces(           std::vector<FH>& faces) const ;
    void sort_faces_contiguous(std::vector<FH>& faces) const ;
    void sort_faces_contiguous_monolithic(std::vector<FH>& faces) const ;

    bool are_contiguous(const FH a, const FH b) const ;

    void dump_faces(           std::vector<FH>& faces) const ;
    void dump_contiguity( const std::vector<FH>& faces ) const ;


    unsigned get_num_boundary(const FH fh) const ;
    bool is_side_or_corner_face(const FH fh) const ;
    bool is_numboundary_face(const FH fh, int numboundary) const ;
    bool is_regular_face(const FH fh, int valence ) const ;
    bool is_interior_face(const FH fh, int margin ) const ;

    typename T::VertexHandle find_vertex_exact( P pt) const ;
    typename T::VertexHandle find_vertex_closest(P pt, float& distance) const ;
    typename T::VertexHandle find_vertex_epsilon(P pt, const float epsilon=-1) const ; // when negative use cfg.epsilon


    T&                      mesh  ;
    const NOpenMeshCfg&     cfg ; 
    NOpenMeshProp<T>&       prop ;
    int                     verbosity ; 
    const nnode*            node ;

    std::vector<NOpenMeshBoundary<T>> loops ; 


};
 


