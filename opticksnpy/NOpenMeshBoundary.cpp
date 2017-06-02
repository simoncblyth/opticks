#include <iostream>

#include "NOpenMeshBoundary.hpp"
#include "NOpenMesh.hpp"


template <typename T>
void NOpenMeshBoundary<T>::CollectLoop( const T* mesh, typename T::HalfedgeHandle start, std::vector<typename T::HalfedgeHandle>& loop)
{
    typedef typename T::HalfedgeHandle      HEH ; 
    HEH heh = start ; 
    do
    {
        if(!mesh->status(heh).deleted()) 
        {
            loop.push_back(heh);                
        } 
        heh = mesh->next_halfedge_handle(heh);
    }
    while( heh != start );
}


template <typename T>
NOpenMeshBoundary<T>::NOpenMeshBoundary( const T* mesh, typename T::HalfedgeHandle start )
{
    CollectLoop( mesh, start, loop );

    std::cout << "NOpenMeshBoundary start " << start << " collected " << loop.size() << " : "  ; 
    for(unsigned i=0 ; i < std::min(unsigned(loop.size()), 10u) ; i++ ) std::cout << " " << loop[i] ;
    std::cout << "..." << std::endl ;         
  

}

template <typename T>
bool NOpenMeshBoundary<T>::contains( typename T::HalfedgeHandle heh )
{
    return std::find(loop.begin(), loop.end(), heh) != loop.end() ;
}
 


template struct NOpenMeshBoundary<NOpenMeshType> ;

