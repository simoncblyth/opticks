#include "PLOG.hh"


#include "NPlaneFromPoints.hpp"
#include "NOpenMeshBoundary.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshZipper.hpp"

#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"



template <typename T>
NOpenMeshZipper<T>::NOpenMeshZipper(
          const NOpenMesh<T>& lhs, 
          const NOpenMesh<T>& rhs
         )
   :
   lhs(lhs), 
   rhs(rhs)
{
    init();
}

template <typename T>
void NOpenMeshZipper<T>::init()
{
    LOG(info) << "NOpenMeshZipper::init"
              << " lhs " << lhs.brief()
              << " rhs " << rhs.brief()
              ;

    dump();
}



template <typename T>
void NOpenMeshZipper<T>::dump()
{
    unsigned n_lhs_inner = lhs.find.inner_loops.size() ;
    unsigned n_rhs_inner = rhs.find.inner_loops.size() ;
    unsigned n_lhs_outer = lhs.find.outer_loops.size() ;
    unsigned n_rhs_outer = rhs.find.outer_loops.size() ;

    std::cout 
         << " n_lhs_inner " << n_lhs_inner
         << " n_rhs_inner " << n_rhs_inner
         << " n_lhs_outer " << n_lhs_outer
         << " n_rhs_outer " << n_rhs_outer
         << std::endl 
         ; 

    for(unsigned i=0 ; i < n_lhs_inner ; i++)  
        dump_boundary( i, lhs.find.inner_loops[i], "lhs_inner" );

    for(unsigned i=0 ; i < n_lhs_outer ; i++)
        dump_boundary( i, lhs.find.outer_loops[i], "lhs_outer" );

    for(unsigned i=0 ; i < n_rhs_inner ; i++)  
        dump_boundary( i, rhs.find.inner_loops[i], "rhs_inner" );

    for(unsigned i=0 ; i < n_rhs_outer ; i++)  
        dump_boundary( i, rhs.find.outer_loops[i], "rhs_outer" );
}



template <typename T>
void NOpenMeshZipper<T>::dump_boundary(int index, const NOpenMeshBoundary<T>& loop, const char* msg)
{
    LOG(info) 
           << msg << " " 
           << std::setw(5) << index  
           << loop.desc()
            ; 

    std::cout << " loop.frontier plane " << loop.frontier.desc() << std::endl ; 
    loop.frontier.dump();

}








template struct NOpenMeshZipper<NOpenMeshType> ;

