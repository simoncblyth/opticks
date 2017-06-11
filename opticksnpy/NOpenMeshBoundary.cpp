#include <iostream>
#include <limits>


#include "PLOG.hh"

#include "NNode.hpp"

#include "NOpenMeshProp.hpp"
#include "NOpenMeshBoundary.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshBisect.hpp"
#include "NOpenMesh.hpp"


template <typename T>
void NOpenMeshBoundary<T>::set_loop_index( int hbl )
{
    for(VHEHI it=loop.begin() ; it != loop.end() ; it++)
    {
        HEH heh = *it ; 
        prop.set_hbloop( heh, hbl ) ;
    }
}


template <typename T>
int NOpenMeshBoundary<T>::get_loop_index()
{
    return prop.get_hbloop( start );
}


template <typename T>
NOpenMeshBoundary<T>::NOpenMeshBoundary(
          T& mesh, 
          const NOpenMeshCfg* cfg, 
          NOpenMeshProp<T>&   prop, 
          HEH start, 
          const nnode* node )
   :
   mesh(mesh),
   cfg(cfg), 
   prop(prop),
   start(start),
   node(node)
{
    init();
}


template <typename T>
void NOpenMeshBoundary<T>::init()
{
    init_sdf();
    init_loop();
    init_range();
    init_frontier();
}

template <typename T>
void NOpenMeshBoundary<T>::init_loop()
{
    HEH heh = start ; 
    do
    {
        if(!mesh.status(heh).deleted()) 
        {
            loop.push_back(heh);                
        } 
        heh = mesh.next_halfedge_handle(heh);
    }
    while( heh != start );
}



template <typename T>
void NOpenMeshBoundary<T>::init_sdf()
{
    assert(node); 
    assert(node->other);

    sdf[0] = node->sdf();
    sdf[1] = node->other->sdf();
}


template <typename T>
void NOpenMeshBoundary<T>::init_range()
{
    range[0] = std::numeric_limits<float>::lowest() ;
    range[1] = std::numeric_limits<float>::max() ;

    for(unsigned i=0 ; i < loop.size() ; i++ )
    {
        HEH h = loop[i] ; 
        VH  v = mesh.to_vertex_handle(h); 
        P p = mesh.point(v);
        float d = signed_distance(COMP_OTHER, p );         
        if( d > range[0] ) range[0] = d ; 
        if( d < range[1] ) range[1] = d ; 
    }

    assert( range[0]*range[1] > 0. && "boundary loop other sdf should be both -ve or both +ve " );
}


template <typename T>
bool NOpenMeshBoundary<T>::is_outer_loop() const 
{
    return range[0] > 0.f && range[1] > 0.f ; 
}
template <typename T>
bool NOpenMeshBoundary<T>::is_inner_loop() const 
{
   return !is_outer_loop();
}


template <typename T>
void NOpenMeshBoundary<T>::init_frontier()
{
    bisect_frontier_edges(frontier, COMP_OTHER);

    frontier_cog[0] = 0. ; 
    frontier_cog[1] = 0. ; 
    frontier_cog[2] = 0. ; 

    for(unsigned i=0 ; i < frontier.size() ; i++)
    {
        frontier_cog += frontier[i] ; 
    }
    frontier_cog /= frontier.size() ;
}
 

template <typename T>
std::string NOpenMeshBoundary<T>::desc(const char* msg, unsigned maxheh) const 
{
    std::stringstream ss ; 
    ss << msg 
       << " halfedge boundary loop "
       << " index " << prop.get_hbloop(start) 
       << " start " << start 
       << " range " << range[0]
       << " -> " << range[1]
       <<  ( is_outer_loop() ? " OUTER " : " INNER " )
       << " num_heh " << loop.size() 
       << " frontier " << frontier.size()
       << " frontier_cog " << NOpenMeshDesc<T>::desc_point( frontier_cog, 8, 2) 
       << " : "  ; 
    for(unsigned i=0 ; i < std::min(unsigned(loop.size()), maxheh) ; i++ ) ss << " " << loop[i] ;
    ss << "..."  ;

    return ss.str();
}

template <typename T>
float NOpenMeshBoundary<T>::signed_distance(NOpenMeshCompType comp, const P& a) const
{
     assert( comp == COMP_THIS || comp == COMP_OTHER );
     return sdf[comp] ? sdf[comp](a[0],a[1],a[2]) : std::numeric_limits<float>::lowest() ;  
}



template <typename T>
std::string NOpenMeshBoundary<T>::fmt(const float f, int w, int p) const 
{
    std::stringstream ss ; 
    ss << " " << std::setw(w) << std::fixed << std::setprecision(p) << f ; 
    return ss.str(); 
}




template <typename T>
void NOpenMeshBoundary<T>::bisect_frontier_edges(std::vector<P>& points, NOpenMeshCompType comp ) const 
{
    //  For leftmesh(rightmesh) edges with comp=LHS(RHS) 
    //  all the signed distances should be close approximations to zero
    //
    //  Need to apply to the other component to get an approximation 
    //  of the analytic frontier.   

    //bool dump(true);
    bool dump(false);

    P pt ;  
    float t ; 

    for(unsigned i=0 ; i < loop.size() ; i++)
    {
        HEH h = loop[i] ; 
        HEH o = mesh.opposite_halfedge_handle(h);   // <-- need to take opposite half edge to descend from the boundary 
        HEH n = mesh.next_halfedge_handle(o) ;   

        bool comp_ok = bisect_frontier_edge(pt, t, n, comp, dump) ;
        if(!comp_ok) break ; 

        assert( t >= 0. && t <= 1. );

        points.push_back(pt) ;
    }

    LOG(info) << "NOpenMeshBoundary<T>::bisect_frontier_edges"
              << " comp " << NOpenMeshEnum::CompType(comp)
              << " points " << points.size()
              ; 

}


template <typename T>
bool  NOpenMeshBoundary<T>::bisect_frontier_edge(P& frontier, float& t, HEH heh, NOpenMeshCompType comp, bool dump) const 
{
    VH    v[2];
    P     p[2];

    v[0] = mesh.from_vertex_handle(heh);   
    v[1] = mesh.to_vertex_handle(heh);   

    p[0] = mesh.point(v[0]);
    p[1] = mesh.point(v[1]);

    float length = (p[1] - p[0]).length() ;
    float tolerance = length/1000. ; 

    NOpenMeshBisect<T> bis(sdf[comp], p[0], p[1], tolerance ); 

    if(bis.degenerate || bis.invalid )
    {
        LOG(warning) << "Cannot bisect : degenerate (both SDF ~zero) or invalid (not bracketing zero)  " ;
        return false ;
    }

    bis.bisect( frontier, t ); 


    if(dump)
    {
        float d[3] ; 
        d[0] = signed_distance( comp, p[0] );  
        d[1] = signed_distance( comp, p[1] );  
        d[2] = bis.func(t);

        std::cout 
             << " h " << std::setw(4) << heh
             << " v[0] " << std::setw(4) << v[0] 
             << " v[1] " << std::setw(4) << v[1] 
             << NOpenMeshDesc<T>::desc_point(p[0],8,2) 
             << NOpenMeshDesc<T>::desc_point(p[1],8,2) 
             << " t " << fmt(t) 
             << " it " << bis.iterations
             << " fr " << NOpenMeshDesc<T>::desc_point(frontier,8,2) 
             ;

        for(unsigned i=0 ; i < 3 ; i++) 
             std::cout 
                  << " d[" << i << "] "
                  <<  fmt(d[i]) 
                  ;

        std::cout << std::endl ; 
    }

    return true ; 
}
 


template <typename T>
void NOpenMeshBoundary<T>::dump(const char* msg, unsigned maxheh) const 
{
    LOG(info) << desc(msg, maxheh) ;

    float sd[2] ; 

    for(unsigned i=0 ; i < loop.size() ; i++)
    {
        HEH heh = loop[i] ; 

        VH tv = mesh.to_vertex_handle(heh);   
        EH eh = mesh.edge_handle(heh);   

        const P tp = mesh.point(tv);

        sd[0] = signed_distance(COMP_THIS, tp );
        sd[1] = signed_distance(COMP_OTHER, tp );

        std::cout 
             << " i " << std::setw(4) << i 
             << " heh " << std::setw(4) << heh
             << " eh " << std::setw(4) << eh
             << " tv " << std::setw(4) << tv 
             << NOpenMeshDesc<T>::desc_point(tp,8,2) 
             ;

        for(unsigned i=0 ; i < 2 ; i++)
        {
            NOpenMeshCompType comp = (NOpenMeshCompType)i ; 
            std::cout 
                 << " sdf_"
                 << NOpenMeshEnum::CompType(comp)
                 << " " << fmt(sd[i]) 
                 ;
        }
        std::cout << std::endl ; 
    }
}




template <typename T>
bool NOpenMeshBoundary<T>::contains( HEH heh ) const 
{
    return std::find(loop.begin(), loop.end(), heh) != loop.end() ;
}

template struct NOpenMeshBoundary<NOpenMeshType> ;

