#include "PLOG.hh"


#include "NOpenMeshBoundary.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshZipper.hpp"

#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"


template <typename T>
NOpenMeshZipper<T>::NOpenMeshZipper(
          const NOpenMeshBoundary<T>& lhs, 
          const NOpenMeshBoundary<T>& rhs
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
}





template struct NOpenMeshZipper<NOpenMeshType> ;

