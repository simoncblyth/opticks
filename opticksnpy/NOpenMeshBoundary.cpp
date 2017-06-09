#include <iostream>

#include "NOpenMeshProp.hpp"
#include "NOpenMeshBoundary.hpp"
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
NOpenMeshBoundary<T>::NOpenMeshBoundary( T& mesh, NOpenMeshProp<T>& prop, HEH start )
   :
   mesh(mesh),
   prop(prop),
   start(start)
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
bool NOpenMeshBoundary<T>::contains( HEH heh )
{
    return std::find(loop.begin(), loop.end(), heh) != loop.end() ;
}

template struct NOpenMeshBoundary<NOpenMeshType> ;

