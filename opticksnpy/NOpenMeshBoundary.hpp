#pragma once

#include "NOpenMeshType.hpp"

template <typename T> struct NOpenMeshProp ; 
struct nnode ; 

/*
NOpenMeshBoundary
==================

Collects vector of halfedges respresenting the mesh boundary 
given a starting halfedge that must lie on the boundary.

Used from NOpenMeshFind<T>::find_boundary_loops


*/ 


template <typename T>
struct NPY_API  NOpenMeshBoundary
{
    typedef typename T::Point              P ; 
    typedef typename T::VertexHandle      VH ; 
    typedef typename T::HalfedgeHandle    HEH ; 
    typedef typename T::EdgeHandle        EH ; 

    typedef std::vector<HEH>              VHEH ; 
    typedef typename VHEH::const_iterator VHEHI ; 

    NOpenMeshBoundary( T& mesh, NOpenMeshProp<T>& prop,  HEH start, const nnode* node );

    void set_loop_index( int hbl );
    int get_loop_index();
    void init();

 
    std::string desc(const char* msg="NOpenMeshBoundary::desc", unsigned maxheh=20u) ;
    void dump(const char* msg="NOpenMeshBoundary::dump", unsigned maxheh=20u) ;
    bool contains(const HEH heh);


    T&                 mesh ; 
    NOpenMeshProp<T>&  prop ; 
    HEH               start ; 
    const nnode*       node ; 

    std::vector<HEH> loop ; 
};
 



