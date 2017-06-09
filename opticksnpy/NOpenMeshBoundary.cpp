#include <iostream>

#include "PLOG.hh"

#include "NNode.hpp"

#include "NOpenMeshProp.hpp"
#include "NOpenMeshBoundary.hpp"
#include "NOpenMeshDesc.hpp"
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
NOpenMeshBoundary<T>::NOpenMeshBoundary( T& mesh, NOpenMeshProp<T>& prop, HEH start, const nnode* node )
   :
   mesh(mesh),
   prop(prop),
   start(start),
   node(node)
{
    init();
}


template <typename T>
void NOpenMeshBoundary<T>::init()
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
std::string NOpenMeshBoundary<T>::desc(const char* msg, unsigned maxheh)
{
    std::stringstream ss ; 
    ss << msg 
       << " halfedge boundary loop "
       << " index " << prop.get_hbloop(start) 
       << " start " << start 
       << " num_heh " << loop.size() 
       << " : "  ; 
    for(unsigned i=0 ; i < std::min(unsigned(loop.size()), maxheh) ; i++ ) ss << " " << loop[i] ;
    ss << "..."  ;

    return ss.str();
}

template <typename T>
void NOpenMeshBoundary<T>::dump(const char* msg, unsigned maxheh)
{
    LOG(info) << desc(msg, maxheh) ;

    unsigned n_heh = loop.size() ; 

    enum { COM, LHS, RHS } ;
    
    float sd[3] ;  
    char labels[3] ; 
    std::function<float(float,float,float)> sdf[3] ; 

    labels[COM] = 'C' ;
    labels[LHS] = 'L' ;
    labels[RHS] = 'R' ;

    if(node)
    {
        sdf[COM] = node->sdf();
        if(node->left && node->right)
        { 
           sdf[LHS] = node->left->sdf();
           sdf[RHS] = node->right->sdf();
        }
    }

    for(unsigned i=0 ; i < n_heh ; i++)
    {
        HEH heh = loop[i] ; 
        VH tv = mesh.to_vertex_handle(heh);   
        EH eh = mesh.edge_handle(heh);   

        const P pt = mesh.point(tv);

        sd[COM] = node        ? sdf[COM](pt[0],pt[1],pt[2]) : 0.f ; 
        sd[LHS] = node->left  ? sdf[LHS](pt[0],pt[1],pt[2]) : 0.f ; 
        sd[RHS] = node->right ? sdf[RHS](pt[0],pt[1],pt[2]) : 0.f ; 

        std::cout 
             << " i " << std::setw(4) << i 
             << " heh " << std::setw(4) << heh
             << " eh " << std::setw(4) << eh
             << " tv " << std::setw(4) << tv 
             << NOpenMeshDesc<T>::desc_point(pt) 
             ;

        for(unsigned i=0 ; i < 3 ; i++)
            std::cout 
                 << " sdf_"
                 << labels[i]
                 << " " << std::setw(10) << std::fixed << std::setprecision(2) << sd[i]
                 ;

        std::cout << std::endl ; 



    }
}




template <typename T>
bool NOpenMeshBoundary<T>::contains( HEH heh )
{
    return std::find(loop.begin(), loop.end(), heh) != loop.end() ;
}

template struct NOpenMeshBoundary<NOpenMeshType> ;

