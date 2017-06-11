#pragma once

#include "NPlaneFromPoints.hpp"
#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"
#include <functional>

struct NOpenMeshCfg ; 
template <typename T> struct NOpenMeshProp ; 
struct nnode ; 

/*
NOpenMeshBoundary
==================

Zero or more boundary loops are collected into a vector or loops 
residing in NOpenMeshFind instances by NOpenMeshFind<T>::find_boundary_loops.
(NOpenMeshFind is a top level constituent of NOpenMesh) 


init_loop
     Collects vector of faceless halfedges respresenting the mesh boundary 
     given a starting halfedge that must be a boundary halfedge 
     (ie without an associated face). The start halfedge is included in
     the loop of halfedges.

init_frontier
     applies bisection root finding to the SDF other function to find
     where the edges cross the analytic frontier, to within a tolerance
     (currently edgelength*0.001 ), the points along the frontier
     are collected


When using copy_faces with PROP_FRONTIER only faces that cross 
sub-object frontiers are included in the mesh. This typically yields
a ring "ribbon" shaped mesh approximating the intersection frontier between the 
sub-objects. 

Frontier ribbon meshes have inner and outer mesh boundary loops, where inner and
outer refer to the SDF depth into the other sub-object.  By construction
the outer SDF values are all +ve and inner loop SDF values all -ve. 
These loops kinda represent the triangulation approximation of the 
actual analytic intersection frontier.

Frontier ribbon mesh::

      +---------+---------+---------+---          outer loop (SDF other >  0)   
       \       / \       / \       / \
      . * . . * . * . . *. .*. . .* . * . . . .   analytic frontier (SDF other = 0 )       
         \   /     \   /     \   /     \
          \ /       \ /       \ /       \
      -----+---------+---------+---------+        inner loop (SDF other < 0) 

*/ 

template <typename T>
struct NPY_API  NOpenMeshBoundary
{
    typedef typename std::function<float(float,float,float)> SDF ; 
    typedef typename T::Point              P ; 
    typedef typename T::VertexHandle      VH ; 
    typedef typename T::HalfedgeHandle    HEH ; 
    typedef typename T::EdgeHandle        EH ; 

    typedef std::vector<HEH>              VHEH ; 
    typedef typename VHEH::const_iterator VHEHI ; 

    NOpenMeshBoundary( T& mesh, 
                       const NOpenMeshCfg* cfg, 
                       NOpenMeshProp<T>&   prop,  
                       HEH start, 
                       const nnode* node );

    void init();
    void init_sdf();
    void init_loop();
    void init_range();
    void init_frontier();

    bool is_outer_loop() const ;
    bool is_inner_loop() const ;
 

    void bisect_frontier_edges() ; 
    bool bisect_frontier_edge(P& p, float& t, HEH heh, NOpenMeshCompType comp, bool dump ) const ;


    std::string fmt(const float f, int w=10, int p=2) const ;
    float signed_distance(NOpenMeshCompType comp, const P& a) const ;


    void set_loop_index( int hbl );
    int get_loop_index();

    std::string desc(const char* msg="NOpenMeshBoundary::desc", unsigned maxheh=20u) const  ;
    void dump(const char* msg="NOpenMeshBoundary::dump", unsigned maxheh=20u) const  ;
    bool contains(const HEH heh) const ;


    T&                   mesh ; 
    const NOpenMeshCfg*  cfg ; 
    NOpenMeshProp<T>&    prop ; 
    HEH                  start ; 
    const nnode*         node ; 

    std::vector<HEH>    loop ; 
    NPlaneFromPoints    frontier ; 

    SDF                 sdf[2]  ; 
    float               range[2] ; 

};


