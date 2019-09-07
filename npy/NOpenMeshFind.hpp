/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
                   const NOpenMeshCfg* cfg,
                   NOpenMeshProp<T>& prop, 
                   const nnode* node);



    void                  dump_boundary_loops(const char* msg="NOpenMeshFind::dump_boundary_loops", bool detail=false) ;
    int                   find_boundary_loops() ;


    bool                    is_selected(const FH fh, NOpenMeshFindType sel, int param) const ;
   

    std::string desc() const ; 
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
    const NOpenMeshCfg*     cfg ; 
    NOpenMeshProp<T>&       prop ;
    int                     verbosity ; 
    const nnode*            node ;

    std::vector<NOpenMeshBoundary<T>> loops ; 
    std::vector<NOpenMeshBoundary<T>> inner_loops ; 
    std::vector<NOpenMeshBoundary<T>> outer_loops ; 


};
 


